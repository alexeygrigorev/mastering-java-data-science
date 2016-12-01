package chapter07.xgb;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import joinery.DataFrame;
import chapter07.cv.Dataset;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;
import com.google.common.collect.Multisets;
import com.google.common.primitives.Doubles;

public class PerformanceDataPreparation {

    public static void main(String[] args) throws IOException {
        prepareData();
    }

    public static Dataset prepareData() throws IOException {
        DataFrame<Object> dataframe = DataFrame.readCsv("data/performance/x_train.csv");

        DataFrame<Object> targetDf = DataFrame.readCsv("data/performance/y_train.csv");
        List<Double> targetList = targetDf.cast(Double.class).col("time");
        double[] target = Doubles.toArray(targetList);

        List<Object> memfreq = noneToNull(dataframe.col("memFreq"));
        List<Object> memtRFC = noneToNull(dataframe.col("memtRFC"));
        dataframe = dataframe.drop("memFreq", "memtRFC");
        dataframe.add("memFreq", memfreq);
        dataframe.add("memtRFC", memtRFC);

        List<Object> types = dataframe.types().stream().map(c -> c.getSimpleName()).collect(Collectors.toList());
        List<Object> columns = new ArrayList<>(dataframe.columns());
        DataFrame<Object> typesDf = new DataFrame<>();
        typesDf.add("column", columns);
        typesDf.add("type", types);

        DataFrame<Object> stringTypes = typesDf.select(p -> p.get(1).equals("String"));
        System.out.println(stringTypes.toString());

        DataFrame<Object> categorical = dataframe.retain(stringTypes.col("column").toArray());
        System.out.println(categorical);

        dataframe = dataframe.drop(stringTypes.col("column").toArray());

        for (Object column : categorical.columns()) {
            List<Object> data = categorical.col(column);
            Multiset<Object> counts = HashMultiset.create(data);
            int nunique = counts.entrySet().size();
            Multiset<Object> countsSorted = Multisets.copyHighestCountFirst(counts);

            System.out.print(column + "\t" + nunique + "\t" + countsSorted);
            List<Object> cleaned = data.stream()
                    .map(o -> counts.count(o) >= 50 ? o : "OTHER")
                    .collect(Collectors.toList());

            dataframe.add(column, cleaned);
        }

        System.out.println(dataframe.head());

        double[][] X = dataframe.toModelMatrix(0.0);
        Dataset dataset = new Dataset(X, target);

        return dataset;
    }

    private static List<Object> noneToNull(List<Object> memfreq) {
        return memfreq.stream()
                .map(s -> isNone(s) ? null : Double.parseDouble(s.toString()))
                .collect(Collectors.toList());
    }

    private static boolean isNone(Object s) {
        return "None".equals(s);
    }
}
