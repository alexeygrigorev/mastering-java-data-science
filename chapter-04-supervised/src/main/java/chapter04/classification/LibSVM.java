package chapter04.classification;

import java.util.List;
import java.util.function.Function;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import chapter04.cv.Dataset;
import chapter04.cv.Fold;
import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;

public class LibSVM {

    public static void mute() {
        svm.svm_set_print_string_function(s -> {});
    }

    public static svm_model train(Dataset dataset, svm_parameter param) {
        svm_problem prob = wrapDataset(dataset);
        return svm.svm_train(prob, param);
    }

    public static DescriptiveStatistics crossValidate(List<Fold> folds, Function<Dataset, svm_model> trainer) {
        double[] aucs = folds.parallelStream().mapToDouble(fold -> {
            Dataset foldTrain = fold.getTrain();
            Dataset foldValidation = fold.getTest();
            svm_model model = trainer.apply(foldTrain);
            return auc(model, foldValidation);
        }).toArray();

        return new DescriptiveStatistics(aucs);
    }

    public static double auc(svm_model model, Dataset dataset) {
        double[] probs = predict(model, dataset);
        return Metrics.auc(dataset.getY(), probs);
    }

    public static double[] predict(svm_model model, Dataset dataset) {
        int n = dataset.length();

        double[][] X = dataset.getX();
        double[] results = new double[n];
        double[] probs = new double[2];

        for (int i = 0; i < n; i++) {
            svm_node[] row = wrapAsSvmNode(X[i]);
            svm.svm_predict_probability(model, row, probs);
            results[i] = probs[1];
        }

        return results;
    }

    public static svm_parameter linearSVC(double C) {
        svm_parameter param = new svm_parameter();
        param.svm_type = svm_parameter.C_SVC;
        param.kernel_type = svm_parameter.LINEAR;
        param.probability = 1;
        param.C = C;

        // defaults
        param.cache_size = 100;
        param.eps = 1e-3;
        param.p = 0.1;
        param.shrinking = 1;
        return param;
    }

    public static svm_parameter polynomialSVC(int degree, double C) {
        svm_parameter param = new svm_parameter();
        param.svm_type = svm_parameter.C_SVC;
        param.kernel_type = svm_parameter.POLY;
        param.C = C;
        param.degree = degree;
        param.gamma = 1;
        param.coef0 = 1;
        param.probability = 1;

        // defaults
        param.cache_size = 100;
        param.eps = 1e-3;
        param.p = 0.1;
        param.shrinking = 1;
        return param;
    }

    public static svm_parameter gaussianSVC(double C, double gamma) {
        svm_parameter param = new svm_parameter();
        param.svm_type = svm_parameter.C_SVC;
        param.kernel_type = svm_parameter.RBF;
        param.C = C;
        param.gamma = gamma;
        param.probability = 1;

        // defaults
        param.cache_size = 100;
        param.eps = 1e-3;
        param.p = 0.1;
        param.shrinking = 1;
        return param;
    }

    private static svm_problem wrapDataset(Dataset dataset) {
        svm_problem prob = new svm_problem();
        prob.l = dataset.length();
        prob.x = wrapAsSvmNodes(dataset.getX());
        prob.y = dataset.getY();
        return prob;
    }

    private static svm_node[][] wrapAsSvmNodes(double[][] X) {
        int n = X.length;
        svm_node[][] nodes = new svm_node[n][];

        for (int i = 0; i < n; i++) {
            nodes[i] = wrapAsSvmNode(X[i]);
        }

        return nodes;
    }

    private static svm_node[] wrapAsSvmNode(double[] dataRow) {
        svm_node[] svmRow = new svm_node[dataRow.length];

        for (int j = 0; j < dataRow.length; j++) {
            svm_node node = new svm_node();
            node.index = j;
            node.value = dataRow[j];
            svmRow[j] = node;
        }

        return svmRow;
    }
}
