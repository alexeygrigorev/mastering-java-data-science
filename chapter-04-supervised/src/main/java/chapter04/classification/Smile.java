package chapter04.classification;

import java.util.List;
import java.util.function.Function;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import chapter04.cv.Dataset;
import chapter04.cv.Fold;
import smile.classification.Classifier;
import smile.validation.AUC;

public class Smile {

    public static DescriptiveStatistics crossValidate(List<Fold> folds,
            Function<Dataset, Classifier<double[]>> trainer) {
        double[] aucs = folds.parallelStream().mapToDouble(fold -> {
            Dataset foldTrain = fold.getTrain();
            Dataset foldValidation = fold.getTest();
            Classifier<double[]> model = trainer.apply(foldTrain);
            return auc(model, foldValidation);
        }).toArray();

        return new DescriptiveStatistics(aucs);
    }

    public static double auc(Classifier<double[]> model, Dataset dataset) {
        double[] probability = predict(model, dataset);
        int[] truth = dataset.getYAsInt();
        return AUC.measure(truth, probability);
    }

    public static double[] predict(Classifier<double[]> model, Dataset dataset) {
        double[][] X = dataset.getX();
        double[] result = new double[X.length];

        for (int i = 0; i < X.length; i++) {
            result[i] = model.predict(X[i]);
        }

        return result;
    }

}
