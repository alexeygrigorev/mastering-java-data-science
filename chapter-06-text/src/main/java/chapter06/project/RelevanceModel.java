package chapter06.project;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Set;

import org.apache.commons.lang3.SerializationUtils;

import chapter06.cv.Dataset;
import chapter06.ml.Metrics;
import joinery.DataFrame;
import smile.classification.DecisionTree.SplitRule;
import smile.classification.RandomForest;
import smile.classification.SoftClassifier;

public class RelevanceModel {

    public static void main(String[] args) throws IOException {
        DataFrame<Number> trainFeatures = load("data/project-train-features.bin");
        DataFrame<Number> testFeatures = load("data/project-test-features.bin");

        Dataset trainDataset = toDataset(trainFeatures);
        Dataset testDataset = toDataset(testFeatures);

        calculateBaselines(testFeatures, testDataset.getY());

        RandomForest rf = new RandomForest.Trainer(100)
                .setMaxNodes(128)
                .setNumRandomFeatures(6)
                .setSamplingRates(0.6)
                .setSplitRule(SplitRule.GINI)
                .train(trainDataset.getX(), trainDataset.getYAsInt());

        double auc = auc(rf, testDataset);
        System.out.println(auc);

        Dataset full = concat(trainDataset, testDataset);
        RandomForest finalModel = new RandomForest.Trainer(100)
                .setMaxNodes(128)
                .setNumRandomFeatures(6)
                .setSamplingRates(0.6)
                .setSplitRule(SplitRule.GINI)
                .train(full.getX(), full.getYAsInt());

        Path path = Paths.get("project/random-forest-model.bin");
        try (OutputStream os = Files.newOutputStream(path)) {
            SerializationUtils.serialize(finalModel, os);
        }
    }

    public static Dataset concat(Dataset d1, Dataset d2) {
        double[][] X = concat(d1.getX(), d2.getX());
        double[] y = concat(d1.getY(), d2.getY());
        return new Dataset(X, y);
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

    private static void calculateBaselines(DataFrame<Number> df, double[] y) {
        df = df.drop("relevance");

        Set<Object> columns = df.columns();
        for (Object columnName : columns) {
            List<Number> col = df.col(columnName);
            double[] baseline = col.stream().mapToDouble(i -> i.doubleValue()).toArray();
            double baselineAuc = Metrics.auc(y, baseline);
            System.out.printf("%s: %.4f%n", String.valueOf(columnName), baselineAuc);
        }
    }

    private static Dataset toDataset(DataFrame<Number> df) {
        double[] y = df.col("relevance").stream().mapToDouble(i -> i.doubleValue()).toArray();
        df = df.drop("relevance");
        double[][] X = df.toModelMatrix(0.0);
        return new Dataset(X, y);
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
