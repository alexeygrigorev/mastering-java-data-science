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
import com.alexeygrigorev.dstools.wrappers.libsvm.LibSVM;
import com.alexeygrigorev.dstools.wrappers.libsvm.LibSVM.Kernel;
import com.alexeygrigorev.dstools.wrappers.libsvm.LibSvmRegressionWrapper;

import chapter04.preprocess.StandardizationPreprocessor;

public class PerformancePredictionLibSVM {

    private static final Metric RMSE = RegressionMetrics.RMSE;

    public static void main(String[] args) throws IOException {
        LibSVM.mute();

        Path path = Paths.get("data/performance.bin");
        Dataset dataset = read(path);

        Split trainTestSplit = dataset.shuffleSplit(0.3);
        Dataset train = trainTestSplit.getTrain();
        Dataset test = trainTestSplit.getTest();

        StandardizationPreprocessor preprocessor = StandardizationPreprocessor.train(train);
        train = preprocessor.transform(train);
        test = preprocessor.transform(test);

        List<Split> folds = train.shuffleKFold(3);

        double[] es = { 0.001, 0.1 };
        double[] Cs = { 0.1, 1.0, 10.0 };
        for (double eps : es) {
            for (double C : Cs) {
                DescriptiveStatistics summary = CV.crossValidate(folds, RMSE, fold -> {
                    LibSvmRegressionWrapper model = LibSVM.epsilonSVR(Kernel.LINEAR, C, eps);
                    model.fit(fold);
                    return model;
                });

                double mean = summary.getMean();
                double std = summary.getStandardDeviation();
                System.out.printf("linear  C=%5.3f, eps=%.3f, rmse=%.4f ± %.4f%n", C, eps, mean, std);
            }
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
