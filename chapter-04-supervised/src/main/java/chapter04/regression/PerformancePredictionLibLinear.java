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

import chapter04.classification.LibLinear;
import chapter04.cv.Dataset;
import chapter04.cv.Split;
import chapter04.preprocess.StandardizationPreprocessor;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.SolverType;
import smile.validation.MSE;

public class PerformancePredictionLibLinear {

    public static void main(String[] args) throws IOException {
        LibLinear.mute();

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
                    Parameter param = new Parameter(SolverType.L2R_L2LOSS_SVR, C, eps);
                    return LibLinear.train(fold, param);
                });

                double mean = summary.getMean();
                double std = summary.getStandardDeviation();
                System.out.printf("svr C=%5.3f, eps=%.3f, rmse=%.4f ± %.4f%n", C, eps, mean, std);
            }
        }
    }

    public static DescriptiveStatistics crossValidate(List<Split> folds,
            Function<Dataset, Model> trainer) {
        double[] aucs = folds.parallelStream().mapToDouble(fold -> {
            Dataset train = fold.getTrain();
            Dataset validation = fold.getTest();
            Model model = trainer.apply(train);
            return rmse(model, validation);
        }).toArray();

        return new DescriptiveStatistics(aucs);
    }

    private static double rmse(Model model, Dataset dataset) {
        double[] prediction = LibLinear.predictValues(model, dataset);
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
