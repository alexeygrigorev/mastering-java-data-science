package chapter04.classification;

import java.util.List;
import java.util.function.Function;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import chapter04.cv.Dataset;
import chapter04.cv.Fold;
import smile.classification.SoftClassifier;
import smile.validation.AUC;

public class Smile {

    public static DescriptiveStatistics crossValidate(List<Fold> folds,
            Function<Dataset, SoftClassifier<double[]>> trainer) {
        double[] aucs = folds.parallelStream().mapToDouble(fold -> {
            Dataset foldTrain = fold.getTrain();
            Dataset foldValidation = fold.getTest();
            SoftClassifier<double[]> model = trainer.apply(foldTrain);
            return auc(model, foldValidation);
        }).toArray();

        return new DescriptiveStatistics(aucs);
    }

    public static double auc(SoftClassifier<double[]> model, Dataset dataset) {
        double[] probability = predict(model, dataset);
        int[] truth = dataset.getYAsInt();
        double auc = AUC.measure(truth, probability);
        if (auc < 0.5) {
            return 1 - auc;
        } else {
            return auc;
        }
    }

    public static double[] predict(SoftClassifier<double[]> model, Dataset dataset) {
        double[][] X = dataset.getX();
        double[] result = new double[X.length];

        double[] probs = new double[2];
        for (int i = 0; i < X.length; i++) {
            model.predict(X[i], probs);
            result[i] = probs[1];
        }

        return result;
    }

}
