package chapter04.regression;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import org.apache.commons.lang3.SerializationUtils;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import com.alexeygrigorev.dstools.cv.CV;
import com.alexeygrigorev.dstools.cv.Split;
import com.alexeygrigorev.dstools.data.Dataset;
import com.alexeygrigorev.dstools.metrics.Metric;
import com.alexeygrigorev.dstools.metrics.RegressionMetrics;
import com.alexeygrigorev.dstools.wrappers.jsat.Jsat;
import com.alexeygrigorev.dstools.wrappers.jsat.JsatRegressionWrapper;
import com.google.common.base.Stopwatch;

import jsat.regression.RidgeRegression;
import jsat.regression.RidgeRegression.SolverMode;

public class PerformancePredictionJSAT {

    private static final Metric RMSE = RegressionMetrics.RMSE;

    public static void main(String[] args) throws IOException {
        Path path = Paths.get("data/performance.bin");
        Dataset dataset = read(path);

        Split trainTestSplit = dataset.shuffleSplit(0.3);
        Dataset train = trainTestSplit.getTrain();

        List<Split> folds = train.shuffleKFold(3);

        double[] lambdas = { 10 };
        for (double lambda : lambdas) {
            Stopwatch stopwatch = Stopwatch.createStarted();

            DescriptiveStatistics summary = CV.crossValidate(folds, RMSE, data -> {
                RidgeRegression ridge = new RidgeRegression();
                ridge.setLambda(lambda);
                ridge.setSolverMode(SolverMode.EXACT_SVD);

                JsatRegressionWrapper model = Jsat.wrap(ridge);
                model.fit(data);
                return model;
            });

            double mean = summary.getMean();
            double std = summary.getStandardDeviation();

            System.out.printf("ridge λ=%9.1f, rmse=%.4f ± %.4f (took %s)%n", lambda, mean, std, stopwatch.stop());
        }
    }

    private static Dataset read(Path path) throws IOException {
        if (!path.toFile().exists()) {
            PerformanceDataPreparation.prepareData();
        }

        try (InputStream is = Files.newInputStream(path)) {
            return SerializationUtils.deserialize(is);
        }
    }

}
