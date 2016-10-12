package chapter05.dimred;

import java.io.IOException;
import java.util.List;
import java.util.function.Function;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import chapter05.cv.Dataset;
import chapter05.cv.Split;
import chapter05.preprocess.StandardizationPreprocessor;
import smile.clustering.KMeans;
import smile.regression.OLS;
import smile.regression.Regression;
import smile.validation.MSE;

public class PerformanceKMeansDimRed {

    public static void main(String[] args) throws IOException {
        Dataset dataset = PerformanceData.readData();

        StandardizationPreprocessor preprocessor = StandardizationPreprocessor.train(dataset);
        dataset = preprocessor.transform(dataset);

        Split split = dataset.trainTestSplit(0.3);
        Dataset train = split.getTrain();

        List<Split> folds = train.shuffleKFold(3);

        int k = 60;
        int maxIter = 10;
        int runs = 1;

        double[] rmses = folds.parallelStream().mapToDouble(fold -> {
            Dataset foldTrain = fold.getTrain();

            double[][] X = foldTrain.getX();
            KMeans km = new KMeans(X, k, maxIter, runs);

            double[][] centroids = km.centroids();
            double[][] distances = distance(X, centroids);
            foldTrain = new Dataset(distances, foldTrain.getY());

            OLS model = new OLS(foldTrain.getX(), foldTrain.getY());

            Dataset validation = fold.getTest();
            double[][] testDistances = distance(validation.getX(), centroids);
            validation = new Dataset(testDistances, validation.getY());

            return rmse(model, validation);
        }).toArray();

        DescriptiveStatistics ols = new DescriptiveStatistics(rmses);
        System.out.printf("ols: rmse=%.4f ± %.4f%n", ols.getMean(), ols.getStandardDeviation());

        double[][] X = train.getX();
        KMeans km = new KMeans(X, k, maxIter, runs);

        double[][] centroids = km.centroids();
        double[][] distances = distance(X, centroids);
        train = new Dataset(distances, train.getY());

        OLS finalModel = new OLS(train.getX(), train.getY());

        Dataset test = split.getTest();
        double[][] testDistancse = distance(test.getX(), centroids);
        test = new Dataset(testDistancse, test.getY());

        double testRmse = rmse(finalModel, test);
        System.out.printf("final model rmse=%.4f%n", testRmse);
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
        double[] rmses = folds.parallelStream().mapToDouble(fold -> {
            Dataset train = fold.getTrain();
            Regression<double[]> model = trainer.apply(train);
            Dataset validation = fold.getTest();
            return rmse(model, validation);
        }).toArray();

        return new DescriptiveStatistics(rmses);
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
