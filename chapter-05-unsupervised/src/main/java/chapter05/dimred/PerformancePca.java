package chapter05.dimred;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import com.google.common.base.Stopwatch;

import chapter05.cv.Dataset;
import chapter05.cv.Split;
import chapter05.preprocess.StandardizationPreprocessor;
import smile.projection.PCA;
import smile.regression.OLS;
import smile.regression.Regression;
import smile.validation.MSE;

public class PerformancePca {

    public static void main(String[] args) throws IOException {
        Dataset dataset = PerformanceData.readData();

        StandardizationPreprocessor preprocessor = StandardizationPreprocessor.train(dataset);
        dataset = preprocessor.transform(dataset);

        Stopwatch stopwatch = Stopwatch.createStarted();
        PCA pca = new PCA(dataset.getX(), false);
        System.out.println("PCA took " + stopwatch.stop());

        double[] variance = pca.getCumulativeVarianceProportion();
        System.out.println(Arrays.toString(variance));

        Split trainTestSplit = dataset.shuffleSplit(0.3);
        Dataset train = trainTestSplit.getTrain();

        List<Split> folds = train.shuffleKFold(3);
        DescriptiveStatistics ols = crossValidate(folds, data -> {
            return new OLS(data.getX(), data.getY());
        });

        System.out.printf("ols: rmse=%.4f ± %.4f%n", ols.getMean(), ols.getStandardDeviation());
        System.out.printf("original dimensionality: %d%n", train.getX()[0].length);

        double[] ratios = { 0.95, 0.99, 0.999 };

        for (double ratio : ratios) {
            pca = pca.setProjection(ratio);
            double[][] projectedX = pca.project(train.getX());
            Dataset projected = new Dataset(projectedX, train.getY());
            System.out.printf("for ratio=%.3f, dimensionality is %d%n", ratio, projectedX[0].length);

            folds = projected.shuffleKFold(3);
            ols = crossValidate(folds, data -> {
                return new OLS(data.getX(), data.getY());
            });

            System.out.printf("ols (%.3f): rmse=%.4f ± %.4f%n", ratio, ols.getMean(), ols.getStandardDeviation());
        }

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
