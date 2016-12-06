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
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.SerializationUtils;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import com.google.common.collect.ImmutableMap;

import chapter07.BeanToJoinery;
import chapter07.Metrics;
import chapter07.RankedPage;
import chapter07.TextUtils;
import chapter07.UrlRepository;
import chapter07.cv.Dataset;
import chapter07.cv.Split;
import joinery.DataFrame;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.IEvaluation;
import ml.dmlc.xgboost4j.java.IObjective;
import ml.dmlc.xgboost4j.java.XGBoost;

public class PageClassificationSimpleFeatures {

    public static void main(String[] args) throws Exception {
        Dataset dataset = readData();

        Map<String, Object> params = XgbUtils.defaultParams();
        params.put("eval_metric", "auc");
        int nrounds = 20;

        Split split = dataset.trainTestSplit(0.2);

        Dataset train = split.getTrain();

        Split trainSplit = train.trainTestSplit(0.2);
        DMatrix dtrain = XgbUtils.wrapData(trainSplit.getTrain());
        DMatrix dval = XgbUtils.wrapData(trainSplit.getTest());
        Map<String, DMatrix> watches = ImmutableMap.of("train", dtrain, "val", dval);

        IObjective obj = null;
        IEvaluation eval = null;

        Booster model;
        model = XGBoost.train(dtrain, params, nrounds, watches, obj, eval);

        List<Split> kfold = train.kfold(3);
        System.out.println("usual CV");

        double aucs = 0;
        for (Split cvSplit : kfold) {
            dtrain = XgbUtils.wrapData(cvSplit.getTrain());
            Dataset validation = cvSplit.getTest();
            dval = XgbUtils.wrapData(validation);

            watches = ImmutableMap.of("train", dtrain, "val", dval);
            model = XGBoost.train(dtrain, params, nrounds, watches, obj, eval);

            double[] predict = XgbUtils.precict(model, dval);

            double auc = Metrics.auc(validation.getY(), predict);

            System.out.printf("fold auc: %.4f%n", auc);
            aucs = aucs + auc;
        }

        aucs = aucs / 3;
        System.out.printf("cv auc: %.4f%n", aucs);

        System.out.println("xgb CV");

        DMatrix dtrainfull = XgbUtils.wrapData(train);
        int nfold = 3;
        String[] metric = {"auc"};
        String[] crossValidation = XGBoost.crossValidation(dtrainfull, params, nrounds, nfold, metric, null, null);

        Arrays.stream(crossValidation).forEach(System.out::println);

        model = XGBoost.train(dtrain, params, nrounds, watches, obj, eval);

        watches = Collections.singletonMap("dtrainfull", dtrainfull);

        // full train
        nrounds = 12;
        model = XGBoost.train(dtrainfull, params, nrounds, watches, obj, eval);

        Dataset test = split.getTest();
        double[] predict = XgbUtils.predict(model, test);
        double auc = Metrics.auc(test.getY(), predict);

        System.out.printf("final auc: %.4f%n", auc);

        Map<String, Integer> importance = XgbUtils.featureImportance(model, dataset.getFeatureNames());
        importance.entrySet().forEach(System.out::println);
    }

    private static Dataset readData() throws IOException {
        File cache = new File("data-cache-text.bin");
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

        List<String> features = JoineryUtils.columnNames(dataframe);
        Dataset dataset = new Dataset(X, target, features);

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
