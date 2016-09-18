package chapter04.classification;

import java.io.IOException;
import java.util.List;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import com.alexeygrigorev.dstools.cv.CV;
import com.alexeygrigorev.dstools.cv.Split;
import com.alexeygrigorev.dstools.data.Dataset;
import com.alexeygrigorev.dstools.metrics.ClassificationMetrics;
import com.alexeygrigorev.dstools.metrics.Metric;
import com.alexeygrigorev.dstools.wrappers.smile.Smile;

import chapter04.RankedPageData;
import chapter04.preprocess.StandardizationPreprocessor;
import smile.classification.DecisionTree.SplitRule;
import smile.classification.GradientTreeBoost;
import smile.classification.LogisticRegression;
import smile.classification.RandomForest;

public class PagePredictionSmile {

    private static final Metric AUC = ClassificationMetrics.AUC;

    public static void main(String[] args) throws IOException {
        Split split = RankedPageData.readRankedPagesMatrix();

        Dataset train = split.getTrain();
        Dataset test = split.getTest();

        StandardizationPreprocessor preprocessor = StandardizationPreprocessor.train(train);
        train = preprocessor.transform(train);
        test = preprocessor.transform(test);

        List<Split> folds = train.kfold(3);

        double[] lambdas = { 0, 0.5, 1.0, 5.0, 10.0, 100.0, 1000.0 };
        for (double lambda : lambdas) {
            DescriptiveStatistics summary = CV.crossValidate(folds, AUC, fold -> {
                LogisticRegression lr = new LogisticRegression.Trainer()
                    .setRegularizationFactor(lambda)
                    .train(fold.getX(), fold.getYAsInt());
                return Smile.wrap(lr);
            });

            double mean = summary.getMean();
            double std = summary.getStandardDeviation();
            System.out.printf("logreg, λ=%8.3f, auc=%.4f ± %.4f%n", lambda, mean, std);
        }

        DescriptiveStatistics rf = CV.crossValidate(folds, AUC, fold -> {
            RandomForest model = new RandomForest.Trainer()
                        .setNumTrees(100)
                        .setNodeSize(4)
                        .setSamplingRates(0.7)
                        .setSplitRule(SplitRule.ENTROPY)
                        .setNumRandomFeatures(3)
                        .train(fold.getX(), fold.getYAsInt());
            return Smile.wrap(model);
        });

        System.out.printf("random forest       auc=%.4f ± %.4f%n", rf.getMean(), rf.getStandardDeviation());

        DescriptiveStatistics gbt = CV.crossValidate(folds, AUC, fold -> {
            GradientTreeBoost model = new GradientTreeBoost.Trainer()
                    .setMaxNodes(100)
                    .setSamplingRates(0.7)
                    .setShrinkage(0.01)
                    .setNumTrees(100)
                    .train(fold.getX(), fold.getYAsInt());
            return Smile.wrap(model);
        });

        System.out.printf("gbt                 auc=%.4f ± %.4f%n", gbt.getMean(), gbt.getStandardDeviation());

        RandomForest rfFinal = new RandomForest.Trainer()
                .setNumTrees(100)
                .setNodeSize(4)
                .setSamplingRates(0.7)
                .setSplitRule(SplitRule.ENTROPY)
                .setNumRandomFeatures(3)
                .train(train.getX(), train.getYAsInt());

        double finalAuc = AUC.evaluate(Smile.wrap(rfFinal), test);
        System.out.printf("final rf            auc=%.4f%n", finalAuc);

        LogisticRegression logregFinal = new LogisticRegression.Trainer()
                .setRegularizationFactor(100.0)
                .train(train.getX(), train.getYAsInt());

        finalAuc = AUC.evaluate(Smile.wrap(logregFinal), test);
        System.out.printf("final logreg        auc=%.4f%n", finalAuc);

    }

}
