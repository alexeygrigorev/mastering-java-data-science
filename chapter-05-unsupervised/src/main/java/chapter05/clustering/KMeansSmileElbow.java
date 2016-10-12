package chapter05.clustering;

import java.io.IOException;
import java.io.PrintWriter;

import com.google.common.base.Stopwatch;

import chapter05.dimred.Categorical;
import chapter05.dimred.Projections;
import chapter05.preprocess.SmileOHE;
import joinery.DataFrame;
import smile.clustering.KMeans;
import smile.data.SparseDataset;
import smile.math.matrix.SingularValueDecomposition;

public class KMeansSmileElbow {

    public static void main(String[] args) throws IOException {
        DataFrame<Object> categorical = Categorical.readData();

        Stopwatch stopwatch = Stopwatch.createStarted();
        SparseDataset sparse = SmileOHE.oneHotEncoding(categorical);
        System.out.println("OHE took " + stopwatch.stop());

        stopwatch = Stopwatch.createStarted();
        SingularValueDecomposition svd = SingularValueDecomposition.decompose(sparse.toSparseMatrix(), 30);
        System.out.println("SVD took " + stopwatch.stop());

        stopwatch = Stopwatch.createStarted();
        double[][] proj = Projections.project(sparse, svd.getV());
        System.out.println("projection took " + stopwatch.stop());

        PrintWriter out = new PrintWriter("distortion.txt");

        for (int k = 3; k < 50; k++) {
            int maxIter = 100;
            int runs = 3;
            KMeans km = new KMeans(proj, k, maxIter, runs);
            out.println(k + "/t" + km.distortion());
        }

        out.close();
    }

}
