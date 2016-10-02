package chapter05.clustering;

import java.io.IOException;

import com.google.common.base.Stopwatch;

import chapter05.dimred.Categorical;
import chapter05.dimred.Projections;
import chapter05.preprocess.OHE;
import joinery.DataFrame;
import smile.data.SparseDataset;

public class CategoricalRandomProjection {

    public static void main(String[] args) throws IOException {
        DataFrame<Object> categorical = Categorical.readData();

        Stopwatch stopwatch = Stopwatch.createStarted();
        SparseDataset sparse = OHE.oneHotEncoding(categorical);
        System.out.println("OHE took " + stopwatch.stop());

        System.out.println("dimensionality: " + sparse.size() + " x " + sparse.ncols());

//        stopwatch = Stopwatch.createStarted();
//        int inputDimension = sparse.ncols();
//        int outputDimension = 100;
//        double[][] basis = Projections.randomProjection(inputDimension, outputDimension, 0);
//        System.out.println("generating random vectors took " + stopwatch.stop());
//        System.out.println("dimensionality is " + inputDimension + " x " + outputDimension);
//
//
//        stopwatch = Stopwatch.createStarted();
//        double[][] proj = Projections.project(sparse, basis);
//        System.out.println("projection took " + stopwatch.stop());
//
        
        
        
    }

}
