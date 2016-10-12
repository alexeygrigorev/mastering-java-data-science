package chapter05.dimred;

import java.io.IOException;
import java.util.List;
import java.util.function.Function;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import com.google.common.base.Stopwatch;

import chapter05.cv.Dataset;
import chapter05.cv.Split;
import chapter05.preprocess.StandardizationPreprocessor;
import smile.clustering.KMeans;
import smile.regression.Regression;
import smile.validation.MSE;

public class DistancesTest {

    public static void main(String[] args) throws IOException {
        Dataset dataset = PerformanceData.readData();

        StandardizationPreprocessor preprocessor = StandardizationPreprocessor.train(dataset);
        dataset = preprocessor.transform(dataset);


        Split split = dataset.trainTestSplit(0.3);
        Dataset train = split.getTrain();

        double[][] X = train.getX();

        int k = 100;
        int maxIter = 100;
        int runs = 3;

        Stopwatch stopwatch = Stopwatch.createStarted();
        KMeans km = new KMeans(X, k, maxIter, runs);
        System.out.println("KMeans took " + stopwatch.stop());

        double[][] centroids = km.centroids();

        stopwatch = Stopwatch.createStarted();
        double[][] distance = distance(X, centroids);
        System.out.println(distance.length);
        System.out.println("Computing distances took " + stopwatch.stop());
    }

    public static double[][] distance(double[][] A, double[][] B) {
        double[] squaredA = squareRows(A);
        double[] squaredB = squareRows(B);

        Array2DRowRealMatrix mA = new Array2DRowRealMatrix(A, false);
        Array2DRowRealMatrix mB = new Array2DRowRealMatrix(B, false);
        double[][] product = mA.multiply(mB.transpose()).getData();

        int nrow = product.length;
        int ncol = product[0].length;
        double[][] distance = new double[nrow][ncol];
        for (int i = 0; i < nrow; i++) {
            for (int j = 0; j < ncol; j++) {
                double dist = squaredA[i] - 2 * product[i][j] + squaredB[j];
                distance[i][j] = Math.sqrt(dist);
            }
        }

        return distance;
    }

    private static double[] squareRows(double[][] data) {
        int nrow = data.length;

        double[] squared = new double[nrow];
        for (int i = 0; i < nrow; i++) {
            double[] row = data[i];

            double res = 0.0;
            for (int j = 0; j < row.length; j++) {
                res = res + row[j] * row[j];
            }

            squared[i] = res;
        }

        return squared;
    }


    public static DescriptiveStatistics crossValidate(List<Split> folds,
            Function<Dataset, Regression<double[]>> trainer) {
        double[] aucs = folds.parallelStream().mapToDouble(fold -> {
            Dataset train = fold.getTrain();
            Dataset validation = fold.getTest();
            Regression<double[]> model = trainer.apply(train);
            return rmse(model, validation);
        }).toArray();

        return new DescriptiveStatistics(aucs);
    }

    private static double rmse(Regression<double[]> model, Dataset dataset) {
        double[] prediction = predict(model, dataset);
        double[] truth = dataset.getY();

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
