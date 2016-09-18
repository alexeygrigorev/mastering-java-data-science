package chapter04.regression;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.function.Function;

import org.apache.commons.lang3.SerializationUtils;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import com.google.common.base.Stopwatch;

import chapter04.cv.Dataset;
import chapter04.cv.Split;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;
import jsat.regression.RidgeRegression;
import jsat.regression.RidgeRegression.SolverMode;
import smile.validation.MSE;

public class PerformancePredictionJSAT {

    public static void main(String[] args) throws IOException {
        Path path = Paths.get("data/performance.bin");
        Dataset dataset = read(path);

        Split trainTestSplit = dataset.shuffleSplit(0.3);
        Dataset train = trainTestSplit.getTrain();

        List<Split> folds = train.shuffleKFold(3);

        DescriptiveStatistics baseline = crossValidate(folds, data -> mean(data));
        System.out.printf("baseline: rmse=%.4f ± %.4f%n", baseline.getMean(), baseline.getStandardDeviation());

        double[] lambdas = { 10 };
        for (double lambda : lambdas) {
            Stopwatch stopwatch = Stopwatch.createStarted();

            DescriptiveStatistics summary = crossValidate(folds, data -> {
                RidgeRegression ridge = new RidgeRegression();
                ridge.setLambda(lambda);
                ridge.setSolverMode(SolverMode.EXACT_SVD);
                ridge.train(data.toJsatRegressionDataset());
                return ridge;
            });

            double mean = summary.getMean();
            double std = summary.getStandardDeviation();

            System.out.printf("ridge λ=%9.1f, rmse=%.4f ± %.4f (took %s)%n", lambda, mean, std, stopwatch.stop());
        }
    }

    public static DescriptiveStatistics crossValidate(List<Split> folds, Function<Dataset, Regressor> trainer) {
        double[] aucs = folds.parallelStream().mapToDouble(fold -> {
            Dataset train = fold.getTrain();
            Dataset validation = fold.getTest();
            Regressor model = trainer.apply(train);
            return rmse(model, validation);
        }).toArray();

        return new DescriptiveStatistics(aucs);
    }

    private static double rmse(Regressor model, Dataset dataset) {
        double[] prediction = predict(model, dataset);
        double[] truth = dataset.getY();

        double mse = new MSE().measure(truth, prediction);
        return Math.sqrt(mse);
    }

    public static double[] predict(Regressor model, Dataset dataset) {
        double[][] X = dataset.getX();
        double[] result = new double[X.length];

        for (int i = 0; i < X.length; i++) {
            DenseVector vector = new DenseVector(X[i]);
            DataPoint point = new DataPoint(vector);
            result[i] = model.regress(point);
        }

        return result;
    }

    private static Dataset read(Path path) throws IOException {
        if (!path.toFile().exists()) {
            PerformanceDataPreparation.prepareData();
        }

        try (InputStream is = Files.newInputStream(path)) {
            return SerializationUtils.deserialize(is);
        }
    }

    private static Regressor mean(Dataset data) {
        double meanTarget = Arrays.stream(data.getY()).average().getAsDouble();
        return new Regressor() {
            @Override
            public void train(RegressionDataSet dataSet, ExecutorService threadPool) {
            }

            @Override
            public void train(RegressionDataSet dataSet) {
            }

            @Override
            public boolean supportsWeightedData() {
                return false;
            }

            @Override
            public double regress(DataPoint data) {
                return meanTarget;
            }

            @Override
            public Regressor clone() {
                try {
                    return (Regressor) super.clone();
                } catch (CloneNotSupportedException e) {
                    throw new RuntimeException(e);
                }
            }
        };
    }

}
