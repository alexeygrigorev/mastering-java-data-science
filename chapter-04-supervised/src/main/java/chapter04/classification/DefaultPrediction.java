package chapter04.classification;

import java.io.IOException;
import java.util.List;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;

import chapter04.cv.Dataset;
import chapter04.cv.Split;
import chapter04.preprocess.StandardizationPreprocessor;
import joinery.DataFrame;
import jsat.classifiers.linear.LogisticRegressionDCD;
import smile.classification.GradientTreeBoost;

public class DefaultPrediction {

    public static void main(String[] args) throws IOException {
        DataFrame<Object> dataframe = DataFrame.readCsv("data/default.csv");
        System.out.println(dataframe.head());

        List<Object> page = dataframe.col("default payment next month");
        double[] target = page.stream().mapToDouble(p -> ((long) p == 0) ? 1.0 : 0.0).toArray();

        List<Object> sex = dataframe.col("SEX");
        System.out.println("sex: " + Sets.newHashSet(sex));
        ImmutableMap<Long, String> sexToString = ImmutableMap.of(1L, "male", 2L, "female");
        sex = Lists.transform(sex, sexToString::get);

        List<Object> educaction = dataframe.col("EDUCATION");
        System.out.println("education: " + Sets.newHashSet(educaction));
        ImmutableMap<Long, String> educationToString = ImmutableMap.of(
                1L, "graduate school", 
                2L, "university", 
                3L, "high school");
        educaction = Lists.transform(educaction, id -> educationToString.getOrDefault(id, "other"));

        List<Object> status = dataframe.col("MARRIAGE");
        System.out.println("status: " + Sets.newHashSet(status));
        ImmutableMap<Long, String> statusToString = ImmutableMap.of(
                1L, "married", 
                2L, "single");
        status = Lists.transform(status, id -> statusToString.getOrDefault(id, "other"));

        dataframe = dataframe.drop("ID", "SEX", "EDUCATION", "MARRIAGE", "default payment next month");
        dataframe.add("sex", sex);
        dataframe.add("educaction", educaction);
        dataframe.add("status", status);

        System.out.println(dataframe.head());

        double[][] X = dataframe.toModelMatrix(0.0);

        Dataset dataset = new Dataset(X, target);
        StandardizationPreprocessor preprocessor = StandardizationPreprocessor.train(dataset);
        dataset = preprocessor.transform(dataset);

        Split split = dataset.shuffleSplit(0.2);
        Dataset train = split.getTrain();
        Dataset test = split.getTest();

        List<Split> folds = train.shuffleKFold(3);

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

        DescriptiveStatistics gbt = Smile.crossValidate(folds, fold -> {
            return new GradientTreeBoost.Trainer()
                    .setMaxNodes(10)
                    .setSamplingRates(0.7)
                    .setShrinkage(0.03)
                    .setNumTrees(400)
                    .train(fold.getX(), fold.getYAsInt());
        });

        System.out.printf("gbt              auc=%.4f ± %.4f%n", gbt.getMean(), gbt.getStandardDeviation());

//        LogisticRegressionDCD finalModel = new LogisticRegressionDCD();
//        finalModel.setC(0.01);
//        finalModel.setMaxIterations(100);
//        finalModel.trainC(train.toJsatDataset());
//
//        double auc = JStat.auc(finalModel, test);
//        System.out.printf("final log reg    auc=%.4f%n", auc);
    }

}
