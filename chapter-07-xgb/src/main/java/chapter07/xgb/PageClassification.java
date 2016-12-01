package chapter07.xgb;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

import joinery.DataFrame;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.IEvaluation;
import ml.dmlc.xgboost4j.java.IObjective;
import ml.dmlc.xgboost4j.java.XGBoost;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.SerializationUtils;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import chapter07.BeanToJoinery;
import chapter07.Metrics;
import chapter07.RankedPage;
import chapter07.TextUtils;
import chapter07.UrlRepository;
import chapter07.cv.Dataset;
import chapter07.cv.Split;

import com.google.common.collect.ImmutableMap;

public class PageClassification {

    public static void main(String[] args) throws Exception {
        Dataset dataset = readData();
        Split split = dataset.trainTestSplit(0.2);

        Dataset trainFull = split.getTrain();
        Dataset test = split.getTest();

        Split trainSplit = trainFull.trainTestSplit(0.2);
        Dataset train = trainSplit.getTrain();
        Dataset val = trainSplit.getTest();
        
        Map<String, Object> params = xgbParams();
        int nrounds = 20;

        DMatrix dtrain = XgbUtils.wrapData(train);
        DMatrix dval = XgbUtils.wrapData(val);
        Map<String, DMatrix> watches = ImmutableMap.of("train", dtrain, "val", dval);

        IObjective obj = null;
        IEvaluation eval = null;
        Booster model = XGBoost.train(dtrain, params, nrounds, watches, obj, eval);

        boolean outputMargin = true;
        int treeLimit = 10;
        float[][] res = model.predict(dval, outputMargin, treeLimit);
        double[] predict = XgbUtils.unwrapToDouble(res);
        double auc = Metrics.auc(val.getY(), predict);
        System.out.printf("auc: %.4f%n", auc);

        System.out.println("usual CV");
        List<Split> kfold = trainFull.kfold(3);

        double aucs = 0;
        for (Split cvSplit : kfold) {
            dtrain = XgbUtils.wrapData(cvSplit.getTrain());
            Dataset validation = cvSplit.getTest();
            dval = XgbUtils.wrapData(validation);

            watches = ImmutableMap.of("train", dtrain, "val", dval);
            model = XGBoost.train(dtrain, params, nrounds, watches, obj, eval);

            predict = XgbUtils.preduct(model, dval);

            auc = Metrics.auc(validation.getY(), predict);

            System.out.printf("fold auc: %.4f%n", auc);
            aucs = aucs + auc;
        }

        aucs = aucs / 3;
        System.out.printf("cv auc: %.4f%n", aucs);

        System.out.println("xgb CV");

        DMatrix dtrainfull = XgbUtils.wrapData(trainFull);
        int nfold = 3;
        String[] metric = {"auc"};
        String[] crossValidation = XGBoost.crossValidation(dtrainfull, params, nrounds, nfold, metric, obj, eval);

        Arrays.stream(crossValidation).forEach(System.out::println);

        model = XGBoost.train(dtrain, params, nrounds, watches, obj, eval);

        // full train
        watches = Collections.singletonMap("dtrainfull", dtrainfull);
        nrounds = 12;
        model = XGBoost.train(dtrainfull, params, nrounds, watches, obj, eval);

        predict = XgbUtils.preduct(model, test);
        auc = Metrics.auc(test.getY(), predict);

        System.out.printf("final auc: %.4f%n", auc);

    }

    public static Map<String, Object> xgbParams() {
        Map<String, Object> params = new HashMap<>();
        params.put("eta", 0.3);
        params.put("gamma", 0);
        params.put("max_depth", 6);
        params.put("min_child_weight", 1);
        params.put("max_delta_step", 0);
        params.put("subsample", 1);
        params.put("colsample_bytree", 1);
        params.put("colsample_bylevel", 1);
        params.put("lambda", 1);
        params.put("alpha", 0);
        params.put("tree_method", "approx");
        params.put("objective", "binary:logistic");
        // params.put("eval_metric", "logloss");
        params.put("eval_metric", "auc");
        params.put("nthread", 8);
        params.put("seed", 42);
        params.put("silent", 1);
        return params;
    }

    private static Dataset readData() throws IOException {
        File cache = new File("data-cache.bin");
        if (cache.exists()) {
            try (InputStream is = Files.newInputStream(cache.toPath())) {
                try (BufferedInputStream bis = new BufferedInputStream(is)) {
                    return SerializationUtils.deserialize(bis);
                }
            }
        }

        UrlRepository urls = new UrlRepository();

        Path path = Paths.get("data/bing-search-results.txt");
        List<String> lines = FileUtils.readLines(path.toFile(), StandardCharsets.UTF_8);

        List<RankedPage> documents = lines.parallelStream().map(line -> parse(urls, line)).filter(Optional::isPresent)
                .map(Optional::get).collect(Collectors.toList());
        urls.close();

        DataFrame<Object> dataframe = BeanToJoinery.convert(documents, RankedPage.class);

        List<Object> page = dataframe.col("page");
        double[] target = page.stream().mapToInt(o -> (int) o).mapToDouble(p -> (p == 0) ? 1.0 : 0.0).toArray();

        dataframe = dataframe.drop("page", "url", "position", "body", "query", "title", "url");
        double[][] X = dataframe.toModelMatrix(0.0);

        Dataset dataset = new Dataset(X, target);

        try (OutputStream os = Files.newOutputStream(cache.toPath())) {
            SerializationUtils.serialize(dataset, os);
        }

        return dataset;
    }

    private static Optional<RankedPage> parse(UrlRepository urls, String line) {
        String[] split = line.split("\t");
        String query = split[0];
        int searchPageNumber = Integer.parseInt(split[1]);
        int position = Integer.parseInt(split[2]);
        String url = split[3];

        Optional<String> html = urls.get(url);
        if (!html.isPresent()) {
            return Optional.empty();
        }

        Document document = Jsoup.parse(html.get());
        if (document.body() == null || document.title() == null) {
            return Optional.empty();
        }

        Element body = document.body();
        List<String> bodyTokens = TextUtils.tokenizeFilter(body.text());
        List<String> titleTokens = TextUtils.tokenizeFilter(document.title());
        List<String> queryTokens = TextUtils.tokenizeFilter(query);

        RankedPage page = new RankedPage(url, position, searchPageNumber);

        page.setQuery(queryTokens);
        page.setTitle(titleTokens);
        page.setBody(bodyTokens);

        int bodyContentLength = body.text().length();
        page.setBodyContentLength(bodyContentLength);

        int numberOfLinks = document.body().select("a").size();
        page.setNumberOfLinks(numberOfLinks);

        Elements headers = body.select("h1,h2,h3,h4,h5,h6");
        int numberOfHeaders = headers.size();
        page.setNumberOfHeaders(numberOfHeaders);

        return Optional.of(page);
    }
}
