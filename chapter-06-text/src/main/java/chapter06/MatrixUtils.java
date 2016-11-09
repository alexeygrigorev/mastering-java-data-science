package chapter06;

import java.util.Arrays;
import java.util.Iterator;

import org.apache.commons.math3.linear.ArrayRealVector;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.sparse.CompRowMatrix;
import no.uib.cipr.matrix.sparse.FlexCompRowMatrix;
import no.uib.cipr.matrix.sparse.SparseVector;
import smile.data.SparseDataset;
import smile.math.SparseArray;
import smile.math.SparseArray.Entry;

public class MatrixUtils {

    public static double[] vectorSimilarity(SparseDataset matrix, SparseArray vector) {
        CompRowMatrix A = asRowMatrix(matrix);
        SparseVector x = asSparseVector(matrix.ncols(), vector);

        DenseVector result = new DenseVector(A.numRows());
        A.mult(x, result);
        return result.getData();
    }

    public static SparseVector asSparseVector(int ncol, SparseArray vector) {
        int size = vector.size();
        int[] indexes = new int[size];
        double[] values = new double[size];

        Iterator<Entry> iterator = vector.iterator();
        int idx = 0;
        while (iterator.hasNext()) {
            Entry entry = iterator.next();
            indexes[idx] = entry.i;
            values[idx] = entry.x;
            idx++;
        }

        return new SparseVector(ncol, indexes, values, false);
    }

    public static double[] vectorSimilarity(double[][] matrix, double[] vector) {
        if (matrix.length > 1_000_000) {
            return slicedSimilarity(matrix, vector, 100_000);
        }

        return mult(matrix, vector);
    }

    private static double[] mult(double[][] matrix, double[] vector) {
        DenseMatrix A = new DenseMatrix(matrix);
        DenseVector x = new DenseVector(vector);

        DenseVector result = new DenseVector(A.numRows());
        A.mult(x, result);
        return result.getData();
    }

    public static double[] slicedSimilarity(double[][] matrix, double[] vector, int sliceSize) {
        int size = matrix.length;
        double[] result = new double[size];

        for (int i = 0; i < size; i = i + sliceSize) {
            int end = i + sliceSize;
            if (end > size) {
                end = size;
            }

            double[][] m = Arrays.copyOfRange(matrix, i, end);
            double[] sim = mult(m, vector);
            System.arraycopy(sim, 0, result, i, sim.length);
        }

        return result;
    }


    public static double[][] matrixSimilarity(SparseDataset d1, SparseDataset d2) {
        CompRowMatrix M1 = asRowMatrix(d1);
        CompRowMatrix M2 = asRowMatrix(d2);

        DenseMatrix M1M2T = new DenseMatrix(M1.numRows(), M2.numRows());
        M1.transBmult(M2, M1M2T);

        return to2d(M1M2T);
    }

    public static double[][] to2d(DenseMatrix dense) {
        double[] data = dense.getData();
        int nrows = dense.numRows();
        int ncols = dense.numColumns();
        double[][] result = new double[nrows][ncols];

        for (int col = 0; col < ncols; col++) {
            for (int row = 0; row < nrows; row++) {
                result[row][col] = data[row + col * nrows];
            }
        }

        return result;
    }

    public static CompRowMatrix asRowMatrix(SparseDataset dataset) {
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

    public static double[][] l2RowNormalize(double[][] data) {
        for (int i = 0; i < data.length; i++) {
            double[] row = data[i];
            ArrayRealVector vector = new ArrayRealVector(row, false);
            double norm = vector.getNorm();
            if (norm != 0) {
                vector.mapDivideToSelf(norm);
                data[i] = vector.getDataRef();
            }
        }

        return data;
    }
}
