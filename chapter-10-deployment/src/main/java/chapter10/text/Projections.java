package chapter10.text;

import java.util.Iterator;

import org.apache.commons.math3.distribution.NormalDistribution;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.sparse.CompRowMatrix;
import no.uib.cipr.matrix.sparse.FlexCompRowMatrix;
import smile.data.SparseDataset;
import smile.math.SparseArray;
import smile.math.SparseArray.Entry;

public class Projections {

    public static double[][] randomProjection(int inputDimension, int outputDimension, int seed) {
        NormalDistribution normal = new NormalDistribution(0.0, 1.0 / outputDimension);
        normal.reseedRandomGenerator(seed);
        double[][] result = new double[inputDimension][];

        for (int i = 0; i < inputDimension; i++) {
            result[i] = normal.sample(outputDimension);
        }

        return result;
    }

    public static double[][] project(double[][] Xd, double[][] Vd) {
        DenseMatrix X = new DenseMatrix(Xd);
        DenseMatrix V = new DenseMatrix(Vd);

        DenseMatrix XV = new DenseMatrix(X.numRows(), V.numColumns());
        X.mult(V, XV);

        return to2d(XV);
    }

    public static double[][] project(SparseDataset dataset, double[][] Vd) {
        CompRowMatrix X = asRowMatrix(dataset);
        DenseMatrix V = new DenseMatrix(Vd);

        DenseMatrix XV = new DenseMatrix(X.numRows(), V.numColumns());
        X.mult(V, XV);

        return to2d(XV);
    }

    public static double[][] to2d(DenseMatrix XV) {
        double[] data = XV.getData();
        int nrows = XV.numRows();
        int ncols = XV.numColumns();
        double[][] result = new double[nrows][ncols];

        for (int col = 0; col < ncols; col++) {
            for (int row = 0; row < nrows; row++) {
                result[row][col] = data[row + col * nrows];
            }
        }

        return result;
    }

    private static CompRowMatrix asRowMatrix(SparseDataset dataset) {
        int ncols = dataset.ncols();
        int nrows = dataset.size();
        FlexCompRowMatrix X = new FlexCompRowMatrix(nrows, ncols);

        SparseArray[] array = dataset.toArray(new SparseArray[0]);
        for (int rowIdx = 0; rowIdx < array.length; rowIdx++) {
            Iterator<Entry> row = array[rowIdx].iterator();
            while (row.hasNext()) {
                Entry entry = row.next();
                X.set(rowIdx, entry.i, entry.x);
            }
        }

        return new CompRowMatrix(X);
    }
}
