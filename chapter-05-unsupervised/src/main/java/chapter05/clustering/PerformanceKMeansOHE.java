package chapter05.clustering;

import java.io.IOException;
import java.util.List;
import java.util.function.Function;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import chapter05.cv.Dataset;
import chapter05.cv.Split;
import chapter05.dimred.PerformanceData;
import chapter05.preprocess.StandardizationPreprocessor;
import smile.clustering.KMeans;
import smile.data.SparseDataset;
import smile.regression.OLS;
import smile.regression.Regression;
import smile.validation.MSE;

public class PerformanceKMeansOHE {

    public static void main(String[] args) throws IOException {
        Dataset dataset = PerformanceData.readData();

        StandardizationPreprocessor preprocessor = StandardizationPreprocessor.train(dataset);
        dataset = preprocessor.transform(dataset);

        Split split = dataset.trainTestSplit(0.3);
        Dataset train = split.getTrain();

        double[][] X = train.getX();

        int k = 60;
        int maxIter = 10;
        int runs = 1;

        KMeans km = new KMeans(X, k, maxIter, runs);
        int[] labels = km.getClusterLabel();
        SparseDataset sparse = oneHotEncoding(labels, k);
        
    }

    private static SparseDataset oneHotEncoding(int[] labels, int k) {
        SparseDataset sparse = new SparseDataset(k);

        for (int i = 0; i < labels.length; i++) {
            sparse.set(i, labels[i], 1.0);
        }

        return sparse;
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
