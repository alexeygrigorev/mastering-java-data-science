package chapter05.clustering;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Multimap;

import chapter05.cv.Dataset;
import chapter05.cv.Split;
import chapter05.dimred.PerformanceData;
import chapter05.preprocess.StandardizationPreprocessor;
import smile.clustering.KMeans;
import smile.regression.Regression;
import smile.validation.MSE;

public class PerformanceKMeansMeanTarget {

    public static void main(String[] args) throws IOException {
        Dataset dataset = PerformanceData.readData();

        System.out.println(dataset.getX().length + " " + dataset.getX()[0].length);

        StandardizationPreprocessor preprocessor = StandardizationPreprocessor.train(dataset);
        dataset = preprocessor.transform(dataset);

        Split split = dataset.trainTestSplit(0.3);
        Dataset train = split.getTrain();

        double[][] X = train.getX();

        int k = 250;
        int maxIter = 10;
        int runs = 1;

        KMeans km = new KMeans(X, k, maxIter, runs);
        double[] y = train.getY();
        int[] labels = km.getClusterLabel();

        Multimap<Integer, Double> groups = ArrayListMultimap.create();
        for (int i = 0; i < labels.length; i++) {
            groups.put(labels[i], y[i]);
        }

        Map<Integer, Double> meanValue = new HashMap<>();
        for (int i = 0; i < k; i++) {
            double mean = groups.get(i).stream().mapToDouble(d -> d).average().getAsDouble();
            meanValue.put(i, mean);
        }

        Dataset test = split.getTest();
        double[][] testX = test.getX();
        int[] testLabels = Arrays.stream(testX).mapToInt(km::predict).toArray();

        double[] testPredict = Arrays.stream(testLabels).mapToDouble(meanValue::get).toArray();
        double result = rmse(test.getY(), testPredict);
        System.out.printf("result rmse: %.4f%n", result);
    }

    private static double rmse(double[] truth, double[] prediction) {
        double mse = new MSE().measure(truth, prediction);
        return Math.sqrt(mse);
    }

    public static double[] predict(Regression<double[]> model, Dataset dataset) {
        double[][] X = dataset.getX();
        double[] result = new double[X.length];

        for (int i = 0; i < X.length; i++) {
            result[i] = model.predict(X[i]);
        }

        return result;
    }
}
