package chapter07.xgb;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import joinery.DataFrame;

public class JoineryUtils {
    public static List<String> columnNames(DataFrame<Object> dataframe) {
        Set<Object> columns = dataframe.columns();
        List<String> results = new ArrayList<>(columns.size());
        for (Object o : columns) {
            results.add(o.toString());
        }
        return results;
    }
}
