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
import jsat.classifiers.linear.LogisticRegressionDCD;
import jsat.classifiers.trees.RandomForest;

public class JStatPagePrediction {

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

        double[] cs = { 0.0001, 0.01, 0.5, 1.0, 5.0, 10.0, 50.0, 70, 100 };
        for (double c : cs) {
            int maxIterations = 100;
            DescriptiveStatistics summary = JStat.crossValidate(folds, fold -> {
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

        DescriptiveStatistics rf = JStat.crossValidate(folds, fold -> {
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

        double auc = JStat.auc(finalModel, test);
        System.out.printf("final log reg    auc=%.4f%n", auc);
    }

}
