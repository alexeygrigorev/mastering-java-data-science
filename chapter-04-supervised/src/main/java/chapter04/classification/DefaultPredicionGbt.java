package chapter04.classification;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.commons.lang3.Validate;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;

import chapter04.cv.Dataset;
import chapter04.cv.Fold;
import joinery.DataFrame;
import joinery.impl.Views;
import smile.classification.GradientTreeBoost;

public class DefaultPredicionGbt {

    public static void main(String[] args) throws IOException {
        DataFrame<Object> dataframe = DataFrame.readCsv("data/default.csv");
        System.out.println(dataframe.head());

        List<Object> page = dataframe.col("default payment next month");
        double[] target = page.stream().mapToDouble(p -> ((long) p == 0) ? 1.0 : 0.0).toArray();

        List<Object> gender = dataframe.col("SEX");
        System.out.println("gender: " + Sets.newHashSet(gender));
        ImmutableMap<Long, String> genderToString = ImmutableMap.of(1L, "male", 2L, "female");
        gender = Lists.transform(gender, genderToString::get);

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

        List<Object> paySum = sumColumns(dataframe, "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6");
        List<Object> billAmtSum = sumColumns(dataframe, "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6");
        List<Object> payAmtSum = sumColumns(dataframe, "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6");

        dataframe = dataframe.drop("ID", "SEX", "EDUCATION", "MARRIAGE", "default payment next month");
        dataframe.add("gender", gender);
        dataframe.add("educaction", educaction);
        dataframe.add("marital_status", status);
        dataframe.add("paySum", paySum);
        dataframe.add("billAmtSum", billAmtSum);
        dataframe.add("payAmtSum", payAmtSum);
        dataframe.add("notPaid", subtract(billAmtSum, payAmtSum));
        dataframe.add("remain", subtract(dataframe.col("LIMIT_BAL"), payAmtSum));

        System.out.println(dataframe.head());

        DataFrame<Number> modelDataFrame = dataframe.toModelMatrixDataFrame();
        double[][] X = modelDataFrame.toModelMatrix(0.0);

        Dataset dataset = new Dataset(X, target);
        Fold split = dataset.shuffleSplit(0.2);
        Dataset allTrain = split.getTrain();
        Dataset test = split.getTest();

        Fold trainSplit = allTrain.shuffleSplit(0.7);
        Dataset train = trainSplit.getTrain();

        int ntrees = 1000;
        System.out.println("training entire tree...");

        GradientTreeBoost gbt;
        gbt = new GradientTreeBoost.Trainer()
                    .setMaxNodes(8)
                    .setSamplingRates(0.6)
                    .setShrinkage(0.03)
                    .setNumTrees(ntrees)
                    .train(train.getX(), train.getYAsInt());

        learningCurves(trainSplit, gbt);

        gbt = new GradientTreeBoost.Trainer()
                .setMaxNodes(8)
                .setSamplingRates(0.6)
                .setShrinkage(0.03)
                .setNumTrees(200)
                .train(train.getX(), train.getYAsInt());

        DataFrame<Object> importances = importanceDataFrame(modelDataFrame, gbt);
        System.out.println(importances.toString(15));
        
        
        System.out.println();
    }

    public static List<Object> sumColumns(DataFrame<Object> df, String... columns) {
        DataFrame<Object> retain = df.retain((Object[]) columns);
        return sumDataFrameColumns(retain);
    }

    public static List<Object> subtract(List<Object> list1, List<Object> list2) {
        Validate.isTrue(list1.size() == list2.size(), "list sizes don't match");
        List<Object> result = new ArrayList<>(list1.size());
        for (int i = 0; i < list1.size(); i++) {
            Number n1 = (Number) list1.get(i);
            Number n2 = (Number) list2.get(i);
            result.add(n1.doubleValue() - n2.doubleValue());
        }

        return result;
    }

    private static List<Object> sumDataFrameColumns(DataFrame<Object> retain) {
        List<List<Object>> listView = new Views.ListView<>(retain, true);
        return listView.stream()
                .map(list -> sumOfElements(list))
                .collect(Collectors.toList());
    }

    private static Object sumOfElements(List<Object> list) {
        return list.stream().mapToDouble(o -> castToDouble(o)).sum();
    }

    private static double castToDouble(Object o) {
        Number number = (Number) o;
        return number.doubleValue();
    }

    private static DataFrame<Object> importanceDataFrame(DataFrame<Number> modelDataFrame, GradientTreeBoost gbt) {
        DataFrame<Object> importances = new DataFrame<>();

        List<Object> columnNames = new ArrayList<>(modelDataFrame.columns());
        importances.add("name", columnNames);

        List<Object> importance = Arrays.stream(gbt.importance()).boxed().collect(Collectors.toList());
        importances.add("importance", importance);

        return importances.sortBy("-importance");
    }

    private static void learningCurves(Fold split, GradientTreeBoost gbt) {
        System.out.println("learning curves...");

        int[] sizes = { 1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50 };

        List<String> toPrint = new ArrayList<String>();
        for (int size : sizes) {
            gbt.trim(size);

            double train = Smile.auc(gbt, split.getTrain());
            double val = Smile.auc(gbt, split.getTest());
            toPrint.add(String.format("%4d: train: %.4f, validation: %.4f", size, train, val));
        }

        Collections.reverse(toPrint);
        toPrint.forEach(System.out::println);
    }

}
