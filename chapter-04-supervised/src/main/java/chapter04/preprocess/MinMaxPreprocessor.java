package chapter04.preprocess;

import java.util.Arrays;

import org.apache.commons.lang3.Validate;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import chapter04.cv.Dataset;

public class MinMaxPreprocessor {

    private final DescriptiveStatistics[] stats;

    public MinMaxPreprocessor(DescriptiveStatistics[] stats) {
        this.stats = stats;
    }

    public static MinMaxPreprocessor train(Dataset dataset) {
        RealMatrix matrix = new Array2DRowRealMatrix(dataset.getX());

        int ncol = matrix.getColumnDimension();
        DescriptiveStatistics[] stats = new DescriptiveStatistics[ncol];

        for (int i = 0; i < ncol; i++) {
            double[] column = matrix.getColumn(i);
            stats[i] = new DescriptiveStatistics(column);
        }

        return new MinMaxPreprocessor(stats);
    }

    public Dataset transform(Dataset dataset) {
        RealMatrix matrix = new Array2DRowRealMatrix(dataset.getX(), true);
        int ncol = matrix.getColumnDimension();
        Validate.isTrue(ncol == stats.length, "wrong shape of input dataset, expected %d columns", stats.length);

        for (int i = 0; i < ncol; i++) {
            double[] column = matrix.getColumn(i);
            double min = stats[i].getMin();
            double max = stats[i].getMax();

            double[] result = Arrays.stream(column).map(d -> (d - min) / (max - min)).toArray();
            matrix.setColumn(i, result);
        }

        return new Dataset(matrix.getData(), dataset.getY());
    }

}
