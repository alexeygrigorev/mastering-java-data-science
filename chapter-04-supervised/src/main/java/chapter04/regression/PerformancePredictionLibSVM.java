package chapter04.regression;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.function.Function;

import org.apache.commons.lang3.SerializationUtils;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import chapter04.classification.LibSVM;
import chapter04.cv.Dataset;
import chapter04.cv.Split;
import chapter04.preprocess.StandardizationPreprocessor;
import libsvm.svm_model;
import libsvm.svm_parameter;
import smile.validation.MSE;

public class PerformancePredictionLibSVM {

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
                DescriptiveStatistics summary = crossValidate(folds, fold -> {
                    svm_parameter param = LibSVM.epsilonSVR(C, eps);
                    return LibSVM.train(fold, param);
                });

                double mean = summary.getMean();
                double std = summary.getStandardDeviation();
                System.out.printf("linear  C=%5.3f, eps=%.3f, rmse=%.4f ± %.4f%n", C, eps, mean, std);
            }
        }
    }

    public static DescriptiveStatistics crossValidate(List<Split> folds,
            Function<Dataset, svm_model> trainer) {
        double[] aucs = folds.parallelStream().mapToDouble(fold -> {
            Dataset train = fold.getTrain();
            Dataset validation = fold.getTest();
            svm_model model = trainer.apply(train);
            return rmse(model, validation);
        }).toArray();

        return new DescriptiveStatistics(aucs);
    }

    private static double rmse(svm_model model, Dataset dataset) {
        double[] prediction = LibSVM.predict(model, dataset);
        double[] truth = dataset.getY();

        double mse = new MSE().measure(truth, prediction);
        return Math.sqrt(mse);
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
