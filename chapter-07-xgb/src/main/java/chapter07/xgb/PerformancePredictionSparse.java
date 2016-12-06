package chapter07.xgb;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;
import com.google.common.collect.Multisets;
import com.google.common.primitives.Doubles;

import joinery.DataFrame;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import smile.data.SparseDataset;

public class PerformancePredictionSparse {

    public static void main(String[] args) throws Exception {
        SparseDataset sparse = readData();

        DMatrix dfull = XgbUtils.wrapData(sparse);

        Map<String, Object> params = XgbUtils.defaultParams();
        params.put("objective", "reg:linear");
        params.put("eval_metric", "rmse");
        int nrounds = 100;

        int nfold = 3;
        String[] metric = {"rmse"};
        String[] crossValidation = XGBoost.crossValidation(dfull, params, nrounds, nfold, metric, null, null);

        Arrays.stream(crossValidation).forEach(System.out::println);

    }


    private static SparseDataset readData() throws IOException {
        DataFrame<Object> dataframe = DataFrame.readCsv("data/performance/x_train.csv");

        DataFrame<Object> targetDf = DataFrame.readCsv("data/performance/y_train.csv");
        List<Double> targetList = targetDf.cast(Double.class).col("time");
        double[] target = Doubles.toArray(targetList);

        dataframe = dataframe.drop("memFreq", "memtRFC");

        List<Object> types = dataframe.types().stream().map(c -> c.getSimpleName()).collect(Collectors.toList());
        List<Object> columns = new ArrayList<>(dataframe.columns());
        DataFrame<Object> typesDf = new DataFrame<>();
        typesDf.add("column", columns);
        typesDf.add("type", types);

        DataFrame<Object> stringTypes = typesDf.select(p -> p.get(1).equals("String"));
        System.out.println(stringTypes.toString());

        DataFrame<Object> categorical = dataframe.retain(stringTypes.col("column").toArray());
        System.out.println(categorical);

        DataFrame<Object> result = new DataFrame<>(dataframe.index(), Collections.emptySet());

        for (Object column : categorical.columns()) {
            List<Object> data = categorical.col(column);
            Multiset<Object> counts = HashMultiset.create(data);
            int nunique = counts.entrySet().size();
            Multiset<Object> countsSorted = Multisets.copyHighestCountFirst(counts);

            System.out.print(column + "\t" + nunique + "\t" + countsSorted);
            List<Object> cleaned = data.stream()
                    .map(o -> counts.count(o) >= 50 ? o : "OTHER")
                    .collect(Collectors.toList());

            result.add(column, cleaned);
        }

        System.out.println(dataframe.head());

        SparseDataset ohe = SmileOHE.oneHotEncoding(result);
        for (int i = 0; i < target.length; i++) {
            ohe.set(i, target[i]);
        }

        return ohe;
    }

}
