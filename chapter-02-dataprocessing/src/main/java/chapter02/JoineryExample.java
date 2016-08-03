package chapter02;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.jooq.lambda.tuple.Tuple2;

import com.aol.cyclops.control.LazyReact;

import joinery.DataFrame;

public class JoineryExample {
    public static void main(String[] args) throws Exception {
        String file = "data/csv-example-generatedata_com.csv";
        DataFrame<Object> df = DataFrame.readCsv(file);

        List<Object> country = df.col("country");
        Map<String, Long> map = LazyReact.sequentialBuilder()
                .from(country)
                .cast(String.class)
                .distinct()
                .zipWithIndex()
                .toMap(Tuple2::v1, Tuple2::v2);

        List<Object> indexes = country.stream().map(map::get).collect(Collectors.toList());
        df = df.drop("country");
        df.add("country_index", indexes);

        System.out.println(df);
    }

}
