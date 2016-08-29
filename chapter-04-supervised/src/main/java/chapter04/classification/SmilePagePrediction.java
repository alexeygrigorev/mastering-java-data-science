package chapter04.classification;

import java.io.IOException;
import java.util.List;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import chapter04.BeanToJoinery;
import chapter04.Data;
import chapter04.RankedPage;
import chapter04.cv.Dataset;
import chapter04.cv.Fold;
import chapter04.preprocess.StandardizationPreprocessor;
import joinery.DataFrame;
import smile.classification.DecisionTree.SplitRule;
import smile.classification.GradientTreeBoost;
import smile.classification.LogisticRegression;
import smile.classification.RandomForest;

public class SmilePagePrediction {

    public static void main(String[] args) throws IOException {
        List<RankedPage> pages = Data.readRankedPages();
        DataFrame<Object> dataframe = BeanToJoinery.convert(pages, RankedPage.class);

        System.out.println(dataframe.head());

        List<Object> page = dataframe.col("page");
        double[] target = page.stream().mapToInt(o -> (int) o).mapToDouble(p -> (p == 0) ? 1.0 : 0.0).toArray();

        dataframe = dataframe.drop("page", "url", "position");
        double[][] X = dataframe.toModelMatrix(0.0);

        Dataset dataset = new Dataset(X, target);
        Fold split = dataset.trainTestSplit(0.2);

        Dataset train = split.getTrain();
        Dataset test = split.getTest();

        StandardizationPreprocessor preprocessor = StandardizationPreprocessor.train(train);
        train = preprocessor.transform(train);
        test = preprocessor.transform(test);

        List<Fold> folds = train.kfold(3);

        double[] lambdas = { 0, 0.5, 1.0, 5.0, 10.0, 100.0 };
        for (double lambda : lambdas) {
            DescriptiveStatistics summary = Smile.crossValidate(folds, fold -> {
                return new LogisticRegression.Trainer()
                        .setRegularizationFactor(lambda)
                        .train(fold.getX(), fold.getYAsInt());
            });

            double mean = summary.getMean();
            double std = summary.getStandardDeviation();
            System.out.printf("logreg, λ=%5.1f, auc=%.4f ± %.4f%n", lambda, mean, std);
        }

        DescriptiveStatistics rf = Smile.crossValidate(folds, fold -> {
            return new RandomForest.Trainer()
                    .setNumTrees(100)
                    .setNodeSize(4)
                    .setSamplingRates(0.7)
                    .setSplitRule(SplitRule.ENTROPY)
                    .setNumRandomFeatures(3)
                    .train(fold.getX(), fold.getYAsInt());
        });

        System.out.printf("random forest    auc=%.4f ± %.4f%n", rf.getMean(), rf.getStandardDeviation());

        DescriptiveStatistics gbt = Smile.crossValidate(folds, fold -> {
            return new GradientTreeBoost.Trainer()
                    .setMaxNodes(100)
                    .setSamplingRates(0.7)
                    .setShrinkage(0.01)
                    .setNumTrees(100)
                    .train(fold.getX(), fold.getYAsInt());
        });

        System.out.printf("gbt              auc=%.4f ± %.4f%n", gbt.getMean(), gbt.getStandardDeviation());

        GradientTreeBoost gbtFinal = new GradientTreeBoost.Trainer()
                .setMaxNodes(100)
                .setSamplingRates(0.7)
                .setShrinkage(0.01)
                .setNumTrees(100)
                .train(train.getX(), train.getYAsInt());

        double finalAuc = Smile.auc(gbtFinal, test);
        System.out.printf("gbt              auc=%.4f", finalAuc);
    }

}
