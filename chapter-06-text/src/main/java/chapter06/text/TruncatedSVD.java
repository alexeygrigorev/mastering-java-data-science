package chapter06.text;

import java.io.Serializable;

import chapter06.MatrixUtils;
import chapter06.Projections;
import smile.data.SparseDataset;
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

    public double[][] transform(SparseDataset data) {
        double[][] result = Projections.project(data, termBasis);
        if (normalize) {
            result = MatrixUtils.l2RowNormalize(result);
        }
        return result;
    }

    public double[][] fitTransform(SparseDataset data) {
        return fit(data).transform(data);
    }

    public double[][] getTermBasis() {
        return termBasis;
    }
}
