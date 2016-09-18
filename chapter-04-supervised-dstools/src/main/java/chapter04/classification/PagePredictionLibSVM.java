package chapter04.classification;

import java.io.IOException;
import java.util.List;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import com.alexeygrigorev.dstools.cv.CV;
import com.alexeygrigorev.dstools.cv.Split;
import com.alexeygrigorev.dstools.data.Dataset;
import com.alexeygrigorev.dstools.metrics.ClassificationMetrics;
import com.alexeygrigorev.dstools.metrics.Metric;
import com.alexeygrigorev.dstools.models.BinaryClassificationModel;
import com.alexeygrigorev.dstools.wrappers.libsvm.LibSVM;

import chapter04.RankedPageData;
import chapter04.preprocess.StandardizationPreprocessor;

public class PagePredictionLibSVM {

    private static final Metric AUC = ClassificationMetrics.AUC;

    public static void main(String[] args) throws IOException {
        Split split = RankedPageData.readRankedPagesMatrix();

        Dataset train = split.getTrain();
        Dataset test = split.getTest();

        StandardizationPreprocessor preprocessor = StandardizationPreprocessor.train(train);
        train = preprocessor.transform(train);
        test = preprocessor.transform(test);

        LibSVM.mute();

        List<Split> folds = train.kfold(3);

        double[] Cs = { 0.001, 0.01, 0.1, 0.5, 1.0, 10.0, 20.0 };
        for (double C : Cs) {
            DescriptiveStatistics summary = CV.crossValidate(folds, AUC, fold -> {
                BinaryClassificationModel model = LibSVM.linearSVC(C);
                model.fit(fold);
                return model;
            });

            double mean = summary.getMean();
            double std = summary.getStandardDeviation();
            System.out.printf("linear  C=%8.3f, auc=%.4f ± %.4f%n", C, mean, std);
        }

        Cs = new double[] { 0.001, 0.01, 0.1, 0.5, 1.0};
        for (double C : Cs) {
            DescriptiveStatistics summary = CV.crossValidate(folds, AUC, fold -> {
                BinaryClassificationModel model = LibSVM.polynomialSVC(2, C);
                model.fit(fold);
                return model;
            });

            double mean = summary.getMean();
            double std = summary.getStandardDeviation();
            System.out.printf("poly(2) C=%8.3f, auc=%.4f ± %.4f%n", C, mean, std);
        }

        Cs = new double[] { 0.001, 0.01, 0.1 };
        for (double C : Cs) {
            DescriptiveStatistics summary = CV.crossValidate(folds, AUC, fold -> {
                BinaryClassificationModel model = LibSVM.polynomialSVC(3, C);
                model.fit(fold);
                return model;
            });

            double mean = summary.getMean();
            double std = summary.getStandardDeviation();
            System.out.printf("poly(3) C=%8.3f, auc=%.4f ± %.4f%n", C, mean, std);
        }

        Cs = new double[] { 0.001, 0.01, 0.1};
        for (double C : Cs) {
            DescriptiveStatistics summary = CV.crossValidate(folds, AUC, fold -> {
                BinaryClassificationModel model = LibSVM.gaussianSVC(C, 1.0);
                model.fit(fold);
                return model;
            });

            double mean = summary.getMean();
            double std = summary.getStandardDeviation();
            System.out.printf("rbf     C=%8.3f, auc=%.4f ± %.4f%n", C, mean, std);
        }

    }

}
