package chapter07.searchengine;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.commons.lang3.SerializationUtils;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.LinkedHashMultiset;
import com.google.common.collect.Multiset;

import chapter07.Metrics;
import chapter07.cv.Dataset;
import chapter07.xgb.JoineryUtils;
import chapter07.xgb.XgbUtils;
import joinery.DataFrame;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.IEvaluation;
import ml.dmlc.xgboost4j.java.IObjective;
import ml.dmlc.xgboost4j.java.XGBoost;
import smile.classification.SoftClassifier;

public class RelevanceModel {

    public static void main(String[] args) throws Exception {
        DataFrame<Number> trainFeatures = load("project/project-train-features.bin");
        System.out.println(JoineryUtils.columnNames(trainFeatures.cast(Object.class)));
        DataFrame<Number> testFeatures = load("project/project-test-features.bin");

        Dataset trainDataset = toDataset(trainFeatures);
        Dataset testDataset = toDataset(testFeatures);

        Map<String, Object> params = XgbUtils.defaultParams();
        params.put("objective", "rank:pairwise");
        params.put("eval_metric", "ndcg@30");
        params.put("colsample_bytree", 0.5);
        params.put("max_depth", 4);
        params.put("min_child_weight", 30);
        params.put("subsample", 0.7);
        params.put("eta", 0.02);

        int nrounds = 500;

        DMatrix dtrain = XgbUtils.wrapData(trainDataset);
        int[] trainGroups = queryGroups(trainFeatures.col("queryId"));
        dtrain.setGroup(trainGroups);

        DMatrix dtest = XgbUtils.wrapData(testDataset);
        int[] testGroups = queryGroups(testFeatures.col("queryId"));
        dtest.setGroup(testGroups);

        Map<String, DMatrix> watches = ImmutableMap.of("train", dtrain, "test", dtest);

        IObjective obj = null;
        IEvaluation eval = null;

        XGBoost.train(dtrain, params, nrounds, watches, obj, eval);

    }

    private static int[] queryGroups(List<Number> queryIdsNumeric) {
        List<Integer> queryIds = queryIdsNumeric.stream().map(Number::intValue).collect(Collectors.toList());
        return groups(queryIds);
    }

    private static int[] groups(List<Integer> queryIds) {
        Multiset<Integer> groupSizes = LinkedHashMultiset.create(queryIds);
        return groupSizes.entrySet().stream().mapToInt(e -> e.getCount()).toArray();
    }

    public static Dataset concat(Dataset d1, Dataset d2) {
        double[][] X = concat(d1.getX(), d2.getX());
        double[] y = concat(d1.getY(), d2.getY());

        return new Dataset(X, y, d1.getFeatureNames());
    }

    public static double[] concat(double[] y1, double[] y2) {
        double[] y = new double[y1.length + y2.length];
        System.arraycopy(y1, 0, y, 0, y1.length);
        System.arraycopy(y2, 0, y, y1.length, y2.length);
        return y;
    }

    public static double[][] concat(double[][] X1, double[][] X2) {
        double[][] X = new double[X1.length + X2.length][];
        System.arraycopy(X1, 0, X, 0, X1.length);
        System.arraycopy(X2, 0, X, X1.length, X2.length);
        return X;
    }

    public static double auc(SoftClassifier<double[]> model, Dataset dataset) {
        double[] probability = predict(model, dataset);
        return Metrics.auc(dataset.getY(), probability);
    }

    public static double[] predict(SoftClassifier<double[]> model, Dataset dataset) {
        double[][] X = dataset.getX();
        double[] result = new double[X.length];

        double[] probs = new double[2];
        for (int i = 0; i < X.length; i++) {
            model.predict(X[i], probs);
            result[i] = probs[1];
        }

        return result;
    }

    private static Dataset toDataset(DataFrame<Number> df) {
        double[] y = df.col("relevance").stream().mapToDouble(i -> i.doubleValue()).toArray();
        df = df.drop("relevance", "queryId");
        double[][] X = df.toModelMatrix(0.0);
        List<String> names = JoineryUtils.columnNames(df.cast(Object.class));
        return new Dataset(X, y, names);
    }

    private static DataFrame<Number> load(String filepath) throws IOException {
        Path path = Paths.get(filepath);
        try (InputStream is = Files.newInputStream(path)) {
            try (BufferedInputStream bis = new BufferedInputStream(is)) {
                DfHolder<Number> holder = SerializationUtils.deserialize(bis);
                return holder.toDf();
            }
        }
    }

}
