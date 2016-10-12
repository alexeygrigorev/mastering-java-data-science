package chapter05.preprocess;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

import joinery.DataFrame;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.SparseVector;

public class JsatOHE {

    public static SimpleDataSet oneHotEncoding(DataFrame<Object> categorical) {
        Map<String, Integer> valueToIndex = new HashMap<>();
        List<Object> columns = new ArrayList<>(categorical.columns());

        int ncol = 0;

        for (Object name : columns) {
            List<Object> column = categorical.col(name);
            Set<Object> distinct = new HashSet<>(column);
            for (Object val : distinct) {
                String stringValue = Objects.toString(name) + "_" + Objects.toString(val);
                valueToIndex.put(stringValue, ncol);
                ncol++;
            }
        }

        int nrows = categorical.length();
        List<DataPoint> points = new ArrayList<>(nrows);

        ListIterator<List<Object>> rows = categorical.iterrows();
        while (rows.hasNext()) {
            List<Object> row = rows.next();

            SparseVector vector = new SparseVector(ncol);

            for (int colIdx = 0; colIdx < columns.size(); colIdx++) {
                Object name = columns.get(colIdx);
                Object val = row.get(colIdx);
                String stringValue = Objects.toString(name) + "_" + Objects.toString(val);
                int targetColIdx = valueToIndex.get(stringValue);

                vector.set(targetColIdx, 1.0);
            }

            points.add(new DataPoint(vector));
        }

        return new SimpleDataSet(points);
    }

}
