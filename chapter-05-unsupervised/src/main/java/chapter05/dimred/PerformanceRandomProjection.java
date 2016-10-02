package chapter05.dimred;

import java.io.IOException;
import java.util.List;
import java.util.function.Function;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import com.google.common.base.Stopwatch;

import chapter05.cv.Dataset;
import chapter05.cv.Split;
import chapter05.preprocess.StandardizationPreprocessor;
import smile.regression.OLS;
import smile.regression.Regression;
import smile.validation.MSE;

public class PerformanceRandomProjection {

    public static void main(String[] args) throws IOException {
        Dataset dataset = PerformanceData.readData();

        StandardizationPreprocessor preprocessor = StandardizationPreprocessor.train(dataset);
        dataset = preprocessor.transform(dataset);

        Stopwatch stopwatch = Stopwatch.createStarted();
        double[][] X = dataset.getX();
        int inputDimension = X[0].length;
        int outputDimension = 100;
        int seed = 1;
        double[][] basis = Projections.randomProjection(inputDimension, outputDimension, seed);
        System.out.println("generating random basis took " + stopwatch.stop());

        double[][] projected = Projections.project(X, basis);
        dataset = new Dataset(projected, dataset.getY());

        Split trainTestSplit = dataset.shuffleSplit(0.3);
        Dataset train = trainTestSplit.getTrain();

        List<Split> folds = train.shuffleKFold(3);
        DescriptiveStatistics ols = crossValidate(folds, data -> {
            return new OLS(data.getX(), data.getY());
        });

        System.out.printf("ols: rmse=%.4f ± %.4f%n", ols.getMean(), ols.getStandardDeviation());
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
