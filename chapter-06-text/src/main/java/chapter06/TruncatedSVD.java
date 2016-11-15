package chapter06;

import smile.data.SparseDataset;
import smile.math.matrix.SingularValueDecomposition;
import smile.math.matrix.SparseMatrix;

public class TruncatedSVD {

    private final int n;
    private final boolean normalize;

    private double[][] rowBasis;

    public TruncatedSVD(int n, boolean normalize) {
        this.n = n;
        this.normalize = normalize;
    }

    public TruncatedSVD fit(SparseDataset data) {
        SparseMatrix matrix = data.toSparseMatrix();
        SingularValueDecomposition svd = SingularValueDecomposition.decompose(matrix, n);
        rowBasis = svd.getV();
        return this;
    }

    public double[][] transform(SparseDataset data) {
        double[][] result = Projections.project(data, rowBasis);
        if (normalize) {
            result = MatrixUtils.l2RowNormalize(result);
        }
        return result;
    }

    public double[][] fitTransform(SparseDataset data) {
        return fit(data).transform(data);
    }
}
