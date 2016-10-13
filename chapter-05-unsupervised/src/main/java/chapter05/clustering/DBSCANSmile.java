package chapter05.clustering;

import java.io.IOException;
import java.util.Random;

import com.google.common.base.Stopwatch;

import chapter05.dimred.Categorical;
import chapter05.dimred.Projections;
import chapter05.preprocess.SmileOHE;
import joinery.DataFrame;
import smile.clustering.DBScan;
import smile.data.SparseDataset;
import smile.math.distance.EuclideanDistance;
import smile.math.matrix.SingularValueDecomposition;

public class DBSCANSmile {

    public static void main(String[] args) throws IOException {
        DataFrame<Object> categorical = Categorical.readData();

        Stopwatch stopwatch = Stopwatch.createStarted();
        SparseDataset sparse = SmileOHE.oneHotEncoding(categorical);
        System.out.println("OHE took " + stopwatch.stop());

        stopwatch = Stopwatch.createStarted();
        SingularValueDecomposition svd = SingularValueDecomposition.decompose(sparse.toSparseMatrix(), 5);
        System.out.println("SVD took " + stopwatch.stop());

        stopwatch = Stopwatch.createStarted();
        double[][] proj = Projections.project(sparse, svd.getV());
        System.out.println("projection took " + stopwatch.stop());

        proj = sample(proj, 10000, 1);

        EuclideanDistance distance = new EuclideanDistance();
        int minPts = 5;
        double radius = 0.2;
        DBScan<double[]> dbscan = new DBScan<>(proj, distance, minPts, radius); 

        System.out.println(dbscan.getNumClusters());
        int[] assignment = dbscan.getClusterLabel();
    }

    private static double[][] sample(double[][] data, int size, long seed) {
        Random rnd = new Random(seed);

        int[] idx = rnd.ints(0, data.length).distinct().limit(size).toArray();
        double[][] sample = new double[size][];
        for (int i = 0; i < size; i++) {
            sample[i] = data[idx[i]];
        }

        return sample;
    }


}
