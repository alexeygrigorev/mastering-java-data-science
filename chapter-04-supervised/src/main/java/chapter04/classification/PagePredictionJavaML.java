package chapter04.classification;

import java.io.IOException;
import java.util.List;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import chapter04.RankedPageData;
import chapter04.cv.Dataset;
import chapter04.cv.Fold;
import chapter04.preprocess.StandardizationPreprocessor;
import net.sf.javaml.classification.tree.RandomForest;

public class PagePredictionJavaML {

    public static void main(String[] args) throws IOException {
        Fold split = RankedPageData.readRankedPagesMatrix();

        Dataset train = split.getTrain();
        Dataset test = split.getTest();

        StandardizationPreprocessor preprocessor = StandardizationPreprocessor.train(train);
        train = preprocessor.transform(train);
        test = preprocessor.transform(test);

        List<Fold> folds = train.kfold(3);

        DescriptiveStatistics rf = JavaML.crossValidate(folds, fold -> randomForest(fold));
        System.out.printf("random forest    auc=%.4f ± %.4f%n", rf.getMean(), rf.getStandardDeviation());
    }

    public static RandomForest randomForest(Dataset train) {
        net.sf.javaml.core.Dataset data = JavaML.wrapDataset(train);

        RandomForest randomForest = new RandomForest(150);
        randomForest.setNumAttributes(3);
        randomForest.buildClassifier(data);

        return randomForest;
    }

}
