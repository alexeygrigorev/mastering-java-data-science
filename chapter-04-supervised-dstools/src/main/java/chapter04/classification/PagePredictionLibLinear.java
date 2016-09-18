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
import com.alexeygrigorev.dstools.wrappers.liblinear.LibLinear;
import com.alexeygrigorev.dstools.wrappers.liblinear.LibLinear.Optimization;
import com.alexeygrigorev.dstools.wrappers.liblinear.LibLinear.Penalty;

import chapter04.RankedPageData;
import chapter04.preprocess.StandardizationPreprocessor;

public class PagePredictionLibLinear {

    private static final Metric AUC = ClassificationMetrics.AUC;

    public static void main(String[] args) throws IOException {
        Split split = RankedPageData.readRankedPagesMatrix();

        Dataset train = split.getTrain();
        Dataset test = split.getTest();

        StandardizationPreprocessor preprocessor = StandardizationPreprocessor.train(train);
        train = preprocessor.transform(train);
        test = preprocessor.transform(test);

        LibLinear.mute();

        List<Split> folds = train.kfold(3);

        double[] Cs = { 0.01, 0.02, 0.03, 0.05, 0.1, 0.5, 1.0, 10.0, 100.0 };
        for (double C : Cs) {
            DescriptiveStatistics summary = CV.crossValidate(folds, AUC, fold -> {
                BinaryClassificationModel model = LibLinear.logisticRegression(C, Penalty.L1,
                        Optimization.PRIMAL);
                model.fit(fold);
                return model;
            });

            double mean = summary.getMean();
            double std = summary.getStandardDeviation();
            System.out.printf("C=%7.3f, auc=%.4f ± %.4f%n", C, mean, std);
        }

        BinaryClassificationModel finalModel = LibLinear.logisticRegression(0.05, Penalty.L1,
                Optimization.PRIMAL);
        finalModel.fit(train);

        double auc = AUC.evaluate(finalModel, test);
        System.out.printf("final log reg auc=%.4f%n", auc);
    }

}
