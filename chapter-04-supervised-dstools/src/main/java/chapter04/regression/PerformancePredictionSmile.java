package chapter04.regression;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang3.SerializationUtils;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import com.alexeygrigorev.dstools.cv.CV;
import com.alexeygrigorev.dstools.cv.Split;
import com.alexeygrigorev.dstools.data.Dataset;
import com.alexeygrigorev.dstools.metrics.Metric;
import com.alexeygrigorev.dstools.metrics.RegressionMetrics;
import com.alexeygrigorev.dstools.models.RegressionModel;
import com.alexeygrigorev.dstools.wrappers.smile.Smile;

import smile.regression.OLS;
import smile.regression.RandomForest;
import smile.regression.Regression;

public class PerformancePredictionSmile {

    private static final Metric RMSE = RegressionMetrics.RMSE;

    public static void main(String[] args) throws IOException {
        Path path = Paths.get("data/performance.bin");
        Dataset dataset = read(path);

        Split trainTestSplit = dataset.shuffleSplit(0.3);
        Dataset train = trainTestSplit.getTrain();
        Dataset test = trainTestSplit.getTest();

        List<Split> folds = train.shuffleKFold(3);

        DescriptiveStatistics baseline = CV.crossValidate(folds, RMSE, data -> mean(data));
        System.out.printf("baseline: rmse=%.4f ± %.4f%n", baseline.getMean(), baseline.getStandardDeviation());

        DescriptiveStatistics ols = CV.crossValidate(folds, RMSE, data -> {
            OLS model = new OLS(data.getX(), data.getY());
            return Smile.wrap(model);
        });

        System.out.printf("ols:      rmse=%.4f ± %.4f%n", ols.getMean(), ols.getStandardDeviation());

        DescriptiveStatistics rf = CV.crossValidate(folds,  RMSE, data -> {
            int nbtrees = 100;
            RandomForest model = new RandomForest.Trainer(nbtrees)
                    .setNumRandomFeatures(15)
                    .setMaxNodes(128)
                    .setNodeSize(10)
                    .setSamplingRates(0.6)
                    .train(data.getX(), data.getY());
            return Smile.wrap(model);
        });
        System.out.printf("rf:       rmse=%.4f ± %.4f%n", rf.getMean(), rf.getStandardDeviation());

        RegressionModel finalOls = Smile.wrap(new OLS(train.getX(), train.getY()));
        double testRmse = RMSE.evaluate(finalOls, test);
        System.out.printf("final rmse=%.4f%n", testRmse);
    }

    private static RegressionModel mean(Dataset data) {
        double meanTarget = Arrays.stream(data.getY()).average().getAsDouble();
        Regression<double[]> smile = x -> meanTarget;
        return Smile.wrap(smile);
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
