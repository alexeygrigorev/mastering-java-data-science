package chapter05.dimred;

import java.io.IOException;

import com.google.common.base.Stopwatch;

import chapter05.preprocess.SmileOHE;
import joinery.DataFrame;
import smile.data.SparseDataset;

public class CategoricalRandomProjection {

    public static void main(String[] args) throws IOException {
        DataFrame<Object> categorical = Categorical.readData();

        Stopwatch stopwatch = Stopwatch.createStarted();
        SparseDataset sparse = SmileOHE.hashingEncoding(categorical, 50_000);
        System.out.println("OHE took " + stopwatch.stop());

        stopwatch = Stopwatch.createStarted();
        double[][] basis = Projections.randomProjection(50_000, 100, 0);
        System.out.println("generating random vectors took " + stopwatch.stop());

        stopwatch = Stopwatch.createStarted();
        double[][] proj = Projections.project(sparse, basis);
        System.out.println("projection took " + stopwatch.stop());
    }

}
