package chapter04.classification;

import java.io.IOException;
import java.util.List;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import chapter04.RankedPageData;
import chapter04.cv.Dataset;
import chapter04.cv.Fold;
import chapter04.preprocess.StandardizationPreprocessor;
import libsvm.svm_model;
import libsvm.svm_parameter;

public class PagePredictionLibSVM {

    public static void main(String[] args) throws IOException {
        Fold split = RankedPageData.readRankedPagesMatrix();

        Dataset train = split.getTrain();
        Dataset test = split.getTest();

        StandardizationPreprocessor preprocessor = StandardizationPreprocessor.train(train);
        train = preprocessor.transform(train);
        test = preprocessor.transform(test);

        LibSVM.mute();

        List<Fold> folds = train.kfold(3);

        double[] Cs = { 0.001, 0.01, 0.1, 0.5, 1.0, 10.0, 20.0 };
        for (double C : Cs) {
            DescriptiveStatistics summary = LibSVM.crossValidate(folds, fold -> {
                svm_parameter param = LibSVM.linearSVC(C);
                svm_model model = LibSVM.train(fold, param);
                return model;
            });

            double mean = summary.getMean();
            double std = summary.getStandardDeviation();
            System.out.printf("linear  C=%8.3f, auc=%.4f ± %.4f%n", C, mean, std);
        }

        Cs = new double[] { 0.001, 0.01, 0.1, 0.5, 1.0};
        for (double C : Cs) {
            DescriptiveStatistics summary = LibSVM.crossValidate(folds, fold -> {
                svm_parameter param = LibSVM.polynomialSVC(2, C);
                svm_model model = LibSVM.train(fold, param);
                return model;
            });

            double mean = summary.getMean();
            double std = summary.getStandardDeviation();
            System.out.printf("poly(2) C=%8.3f, auc=%.4f ± %.4f%n", C, mean, std);
        }

        Cs = new double[] { 0.001, 0.01, 0.1 };
        for (double C : Cs) {
            DescriptiveStatistics summary = LibSVM.crossValidate(folds, fold -> {
                svm_parameter param = LibSVM.polynomialSVC(3, C);
                return LibSVM.train(fold, param);
            });

            double mean = summary.getMean();
            double std = summary.getStandardDeviation();
            System.out.printf("poly(3) C=%8.3f, auc=%.4f ± %.4f%n", C, mean, std);
        }

        Cs = new double[] { 0.001, 0.01, 0.1};
        for (double C : Cs) {
            DescriptiveStatistics summary = LibSVM.crossValidate(folds, fold -> {
                svm_parameter param = LibSVM.gaussianSVC(C, 1.0);
                svm_model model = LibSVM.train(fold, param);
                return model;
            });

            double mean = summary.getMean();
            double std = summary.getStandardDeviation();
            System.out.printf("rbf     C=%8.3f, auc=%.4f ± %.4f%n", C, mean, std);
        }

    }

}
