package chapter07.text;

import java.io.Serializable;

import chapter07.cv.Dataset;
import smile.data.SparseDataset;
import smile.math.matrix.Matrix;
import smile.math.matrix.SingularValueDecomposition;
import smile.math.matrix.SparseMatrix;

public class TruncatedSVD implements Serializable {

    private final int n;
    private final boolean normalize;

    private double[][] termBasis;

    public TruncatedSVD(int n, boolean normalize) {
        this.n = n;
        this.normalize = normalize;
    }

    public TruncatedSVD fit(SparseDataset data) {
        SparseMatrix matrix = data.toSparseMatrix();
        SingularValueDecomposition svd = SingularValueDecomposition.decompose(matrix, n);
        termBasis = svd.getV();
        return this;
    }

    public TruncatedSVD fit(Dataset data) {
        Matrix matrix = new Matrix(data.getX());
        SingularValueDecomposition svd = SingularValueDecomposition.decompose(matrix, n);
        termBasis = svd.getV();
        return this;
    }

    public double[][] transform(SparseDataset data) {
        double[][] result = Projections.project(data, termBasis);
        if (normalize) {
            result = MatrixUtils.l2RowNormalize(result);
        }
        return result;
    }

    public double[][] transform(Dataset data) {
        double[][] result = Projections.project(data.getX(), termBasis);
        if (normalize) {
            result = MatrixUtils.l2RowNormalize(result);
        }
        return result;
    }

    public double[][] fitTransform(SparseDataset data) {
        return fit(data).transform(data);
    }

    public double[][] fitTransform(Dataset data) {
        return fit(data).transform(data);
    }

    public double[][] getTermBasis() {
        return termBasis;
    }

    public int getN() {
        return n;
    }
}
