package chapter04.classification;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import chapter04.cv.Dataset;
import chapter04.cv.Fold;
import net.sf.javaml.classification.Classifier;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.DenseInstance;
import net.sf.javaml.core.Instance;
import smile.validation.AUC;

public class JavaML {

    public static DescriptiveStatistics crossValidate(List<Fold> folds, Function<Dataset, Classifier> trainer) {
        double[] aucs = folds.parallelStream().mapToDouble(fold -> {
            Dataset foldTrain = fold.getTrain();
            Dataset foldValidation = fold.getTest();
            Classifier model = trainer.apply(foldTrain);
            return auc(model, foldValidation);
        }).toArray();

        return new DescriptiveStatistics(aucs);
    }

    public static double auc(Classifier model, Dataset dataset) {
        double[] probability = predict(model, dataset);
        int[] truth = dataset.getYAsInt();
        return AUC.measure(truth, probability);
    }

    public static double[] predict(Classifier model, Dataset dataset) {
        double[][] X = dataset.getX();
        double[] result = new double[X.length];

        for (int i = 0; i < X.length; i++) {
            DenseInstance point = new DenseInstance(X[i]);
            Map<Object, Double> distribution = model.classDistribution(point);
            result[i] = distribution.get(1);
        }

        return result;
    }

    public static net.sf.javaml.core.Dataset wrapDataset(Dataset train) {
        double[][] X = train.getX();
        int[] y = train.getYAsInt();

        List<Instance> rows = new ArrayList<>(X.length);
        for (int i = 0; i < X.length; i++) {
            rows.add(new DenseInstance(X[i], y[i]));
        }

        return new DefaultDataset(rows);
    }
}
