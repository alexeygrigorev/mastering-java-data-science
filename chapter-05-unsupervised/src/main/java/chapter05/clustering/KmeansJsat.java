package chapter05.clustering;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;

import com.google.common.base.Stopwatch;
import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multimap;
import com.google.common.collect.Multiset;
import com.google.common.collect.Multiset.Entry;
import com.google.common.collect.Multisets;
import com.google.common.collect.Ordering;

import chapter05.dimred.Categorical;
import chapter05.dimred.Projections;
import chapter05.preprocess.JsatOHE;
import chapter05.preprocess.SmileOHE;
import joinery.DataFrame;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.clustering.SeedSelectionMethods.SeedSelection;
import jsat.clustering.kmeans.ElkanKMeans;
import jsat.clustering.kmeans.HamerlyKMeans;
import jsat.clustering.kmeans.KMeans;
import jsat.linear.DenseVector;
import jsat.linear.distancemetrics.EuclideanDistance;
import smile.data.SparseDataset;
import smile.math.matrix.SingularValueDecomposition;

public class KmeansJsat {

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

        SimpleDataSet data = wrap(proj);

        stopwatch = Stopwatch.createStarted();
        KMeans km = new HamerlyKMeans(new EuclideanDistance(), SeedSelection.RANDOM, new Random(1));
        List<List<DataPoint>> clustering = km.cluster(data);
        System.out.println("kmeans took " + stopwatch.stop());
    }

    private static SimpleDataSet wrap(double[][] proj) {
        int nrows = proj.length;
        List<DataPoint> points = new ArrayList<>(nrows);

        for (int i = 0 ; i < nrows; i++) {
            DenseVector vector = new DenseVector(proj[i]);
            points.add(new DataPoint(vector));
        }

        return new SimpleDataSet(points);
    }

    private static void printValues(Multimap<Integer, String> map) {
        List<Integer> keys = Ordering.natural().sortedCopy(map.keySet());

        for (Integer c : keys) {
            System.out.print(c + ": ");

            Collection<String> values = map.get(c);
            Multiset<String> counts = HashMultiset.create(values);
            counts = Multisets.copyHighestCountFirst(counts);

            int totalSize = values.size();
            for (Entry<String> e : counts.entrySet()) {
                double ratio = 1.0 * e.getCount() / totalSize;
                String element = e.getElement();
                System.out.printf("%s=%.3f (%d), ", element, ratio, e.getCount());
            }

            System.out.println();
        }
    }

}
