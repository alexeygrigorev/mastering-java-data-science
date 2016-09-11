package chapter04.regression;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

import org.apache.commons.lang3.SerializationUtils;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import com.google.common.base.Stopwatch;

import chapter04.cv.Dataset;
import chapter04.cv.Split;
import smile.regression.LASSO;
import smile.regression.Regression;
import smile.validation.MSE;

public class PerformancePrediction {

    public static void main(String[] args) throws IOException {
        Path path = Paths.get("data/performance.bin");
        Dataset dataset = read(path);

        Split trainTestSplit = dataset.shuffleSplit(0.3);
        Dataset train = trainTestSplit.getTrain();
        Dataset test = trainTestSplit.getTest();

        List<Split> folds = train.shuffleKFold(3);

        DescriptiveStatistics baseline = crossValidate(folds, data -> mean(data));
        System.out.printf("baseline: rmse=%.4f ± %.4f%n", baseline.getMean(), baseline.getStandardDeviation());

        double[] lambdas = { 0.1, 1, 10, 100, 1000, 5000, 10000, 20000 };
        for (double lambda : lambdas) {
            Stopwatch stopwatch = Stopwatch.createStarted();

            DescriptiveStatistics summary = crossValidate(folds, data -> {
                return new LASSO(data.getX(), data.getY(), lambda);
            });

            double mean = summary.getMean();
            double std = summary.getStandardDeviation();

            System.out.printf("lasso λ=%9.1f, rmse=%.4f ± %.4f (took %s)%n", 
                    lambda, mean, std, stopwatch.stop());
        }

        LASSO lasso = new LASSO(train.getX(), train.getY(), 10000);
        double testRmse = rmse(lasso, test);
        System.out.printf("final rmse=%.4f%n", testRmse);
    }

    private static Regression<double[]> mean(Dataset data) {
        double meanTarget = Arrays.stream(data.getY()).average().getAsDouble();
        return x -> meanTarget;
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

    private static Dataset read(Path path) throws IOException {
        try (InputStream is = Files.newInputStream(path)) {
            return SerializationUtils.deserialize(is);
        }
    }

}
