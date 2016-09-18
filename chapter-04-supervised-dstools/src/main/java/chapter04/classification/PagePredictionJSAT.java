package chapter04.classification;

import java.io.IOException;
import java.util.List;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import com.alexeygrigorev.dstools.cv.CV;
import com.alexeygrigorev.dstools.cv.Split;
import com.alexeygrigorev.dstools.data.Dataset;
import com.alexeygrigorev.dstools.metrics.ClassificationMetrics;
import com.alexeygrigorev.dstools.metrics.Metric;
import com.alexeygrigorev.dstools.wrappers.jsat.Jsat;
import com.alexeygrigorev.dstools.wrappers.jsat.JsatBinaryClassificationWrapper;

import chapter04.RankedPageData;
import chapter04.preprocess.StandardizationPreprocessor;
import jsat.classifiers.linear.LogisticRegressionDCD;
import jsat.classifiers.svm.SBP;
import jsat.classifiers.svm.SupportVectorLearner.CacheMode;
import jsat.classifiers.trees.RandomForest;
import jsat.distributions.kernels.KernelTrick;
import jsat.distributions.kernels.PolynomialKernel;
import jsat.regression.LogisticRegression;

public class PagePredictionJSAT {

    private static final Metric AUC = ClassificationMetrics.AUC;

    public static void main(String[] args) throws IOException {
        Split split = RankedPageData.readRankedPagesMatrix();

        Dataset train = split.getTrain();
        Dataset test = split.getTest();

        StandardizationPreprocessor preprocessor = StandardizationPreprocessor.train(train);
        train = preprocessor.transform(train);
        test = preprocessor.transform(test);

        List<Split> folds = train.kfold(3);

        DescriptiveStatistics logreg = CV.crossValidate(folds, AUC, fold -> {
            JsatBinaryClassificationWrapper model = Jsat.wrapClassifier(new LogisticRegression());
            model.fit(fold);
            return model;
        });

        System.out.printf("plain logreg     auc=%.4f ± %.4f%n", logreg.getMean(), logreg.getStandardDeviation());

        double[] cs = { 0.0001, 0.01, 0.5, 1.0, 5.0, 10.0, 50.0, 70, 100 };
        for (double c : cs) {
            int maxIterations = 100;
            DescriptiveStatistics summary = CV.crossValidate(folds, AUC, fold -> {
                LogisticRegressionDCD model = new LogisticRegressionDCD();
                model.setMaxIterations(maxIterations);
                model.setC(c);

                JsatBinaryClassificationWrapper wrapper = Jsat.wrap(model);
                wrapper.fit(fold);
                return wrapper;
            });

            double mean = summary.getMean();
            double std = summary.getStandardDeviation();
            System.out.printf("logreg, C=%5.1f, auc=%.4f ± %.4f%n", c, mean, std);
        }

        KernelTrick kernel = new PolynomialKernel(2);
        CacheMode cacheMode = CacheMode.FULL;

        double[] nus = { 0.3, 0.5, 0.7 };
        for (double nu : nus) {
            int maxIterations = 30;
            DescriptiveStatistics summary = CV.crossValidate(folds, AUC, fold -> {
                SBP sbp = new SBP(kernel, cacheMode, maxIterations, nu);
                JsatBinaryClassificationWrapper wrapper = Jsat.wrap(sbp);
                wrapper.fit(fold);
                return wrapper;
            });

            double mean = summary.getMean();
            double std = summary.getStandardDeviation();
            System.out.printf("sbp    nu=%5.1f, auc=%.4f ± %.4f%n", nu, mean, std);
        }

        DescriptiveStatistics rf = CV.crossValidate(folds, AUC, fold -> {
            RandomForest model = new RandomForest();
            model.setFeatureSamples(4);
            model.setMaxForestSize(150);
            JsatBinaryClassificationWrapper wrapper = Jsat.wrapClassifier(model);
            wrapper.fit(fold);
            return wrapper;
        });

        System.out.printf("random forest    auc=%.4f ± %.4f%n", rf.getMean(), rf.getStandardDeviation());

        JsatBinaryClassificationWrapper finalModel = Jsat.wrapClassifier(new LogisticRegression());
        finalModel.fit(train);

        double auc = AUC.evaluate(finalModel, test);
        System.out.printf("final log reg auc=%.4f%n", auc);
    }

}
