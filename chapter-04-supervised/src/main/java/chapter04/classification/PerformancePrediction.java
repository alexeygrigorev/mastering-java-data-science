package chapter04.classification;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import com.google.common.primitives.Doubles;

import joinery.DataFrame;

public class PerformancePrediction {

    public static void main(String[] args) throws IOException {
        DataFrame<Object> dataframe = DataFrame.readCsv("data/performance/x_train.csv");

        DataFrame<Object> targetDf = DataFrame.readCsv("data/performance/y_train.csv");
        List<Double> targetList = targetDf.cast(Double.class).col("time");
        double[] target = Doubles.toArray(targetList);

        List<Object> types = dataframe.types().stream().map(c -> c.getSimpleName()).collect(Collectors.toList());
        List<Object> columns = new ArrayList<>(dataframe.columns());
        DataFrame<Object> typesDf = new DataFrame<>();
        typesDf.add("column", columns);
        typesDf.add("type", types);

        DataFrame<Object> stringTypes = typesDf.select(p -> p.get(1).equals("String"));
        System.out.println(stringTypes.toString(1000));
        
    }
}
