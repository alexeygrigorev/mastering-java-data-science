package chapter05.dimred;

import java.io.IOException;

import com.google.common.base.Stopwatch;

import chapter05.preprocess.OHE;
import joinery.DataFrame;
import smile.data.SparseDataset;
import smile.math.matrix.SingularValueDecomposition;
import smile.math.matrix.SparseMatrix;

public class CategoricalSvd {

    public static void main(String[] args) throws IOException {
        DataFrame<Object> categorical = Categorical.readData();

        Stopwatch stopwatch = Stopwatch.createStarted();
        SparseDataset sparse = OHE.hashingEncoding(categorical, 50_000);
        System.out.println("OHE took " + stopwatch.stop());

        stopwatch = Stopwatch.createStarted();
        SparseMatrix matrix = sparse.toSparseMatrix();
        SingularValueDecomposition svd = SingularValueDecomposition.decompose(matrix, 100);
        System.out.println("SVD took " + stopwatch.stop());

        System.out.println("V dim: " + svd.getV().length + " x " + svd.getV()[0].length);

        stopwatch = Stopwatch.createStarted();
        double[][] proj = Projections.project(sparse, svd.getV());
        System.out.println("projection: " + proj.length + " x " + proj[0].length + ", " + stopwatch.stop());
    }

}