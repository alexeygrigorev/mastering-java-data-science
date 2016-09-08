package chapter04.classification;

import java.io.IOException;
import java.util.List;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import chapter04.RankedPageData;
import chapter04.cv.Dataset;
import chapter04.cv.Fold;
import chapter04.preprocess.StandardizationPreprocessor;
import jsat.classifiers.linear.LogisticRegressionDCD;
import jsat.classifiers.trees.RandomForest;

public class PagePredictionJSAT {

    public static void main(String[] args) throws IOException {
        Fold split = RankedPageData.readRankedPagesMatrix();

        Dataset train = split.getTrain();
        Dataset test = split.getTest();

        StandardizationPreprocessor preprocessor = StandardizationPreprocessor.train(train);
        train = preprocessor.transform(train);
        test = preprocessor.transform(test);

        List<Fold> folds = train.kfold(3);

        double[] cs = { 0.0001, 0.01, 0.5, 1.0, 5.0, 10.0, 50.0, 70, 100 };
        for (double c : cs) {
            int maxIterations = 100;
            DescriptiveStatistics summary = JSAT.crossValidate(folds, fold -> {
                LogisticRegressionDCD model = new LogisticRegressionDCD();
                model.setMaxIterations(maxIterations);
                model.setC(c);
                model.trainC(fold.toJsatDataset());
                return model; 
            });

            double mean = summary.getMean();
            double std = summary.getStandardDeviation();
            System.out.printf("logreg, C=%5.1f, auc=%.4f ± %.4f%n", c, mean, std);
        }

        DescriptiveStatistics rf = JSAT.crossValidate(folds, fold -> {
            RandomForest model = new RandomForest();
            model.setFeatureSamples(4);
            model.setMaxForestSize(150);
            model.trainC(fold.toJsatDataset());
            return model;
        });

        System.out.printf("random forest    auc=%.4f ± %.4f%n", rf.getMean(), rf.getStandardDeviation());

        LogisticRegressionDCD finalModel = new LogisticRegressionDCD();
        finalModel.setC(0.0001);
        finalModel.setMaxIterations(100);
        finalModel.trainC(train.toJsatDataset());

        double auc = JSAT.auc(finalModel, test);
        System.out.printf("final log reg    auc=%.4f%n", auc);
    }

}
