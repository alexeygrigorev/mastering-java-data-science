package chapter05.clustering;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import com.google.common.base.Stopwatch;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multimap;
import com.google.common.collect.Multiset;
import com.google.common.collect.Multiset.Entry;
import com.google.common.collect.Multisets;
import com.google.common.collect.Ordering;

import chapter05.dimred.Categorical;
import chapter05.dimred.Projections;
import chapter05.preprocess.SmileOHE;
import joinery.DataFrame;
import smile.clustering.XMeans;
import smile.data.SparseDataset;
import smile.math.matrix.SingularValueDecomposition;

public class XmeansSmile {

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

        stopwatch = Stopwatch.createStarted();
        int kmax = 3000;
        XMeans km = new XMeans(proj, kmax);
        System.out.println("XMeans took " + stopwatch.stop());
        System.out.println("selected number of clusters: " + km.getNumClusters());
        System.out.println(Arrays.toString(km.getClusterSize()));

        int[] assignment = km.getClusterLabel();

        DataFrame<Object> data = Categorical.readRestOfData();

        List<Object> resp = data.col("company_response_to_consumer");
        List<Object> timely = data.col("timely_response");
        List<Object> disputed = data.col("consumer_disputed?");

        Multimap<Integer, String> respMap = ArrayListMultimap.create();
        Multimap<Integer, String> timelyMap = ArrayListMultimap.create();
        Multimap<Integer, String> disputedMap = ArrayListMultimap.create();

        for (int i = 0; i < assignment.length; i++) {
            int cluster = assignment[i];
            respMap.put(cluster, resp.get(i).toString());
            timelyMap.put(cluster, timely.get(i).toString());
            disputedMap.put(cluster, disputed.get(i).toString());
        }

        System.out.println("company_response_to_consumer");
        printValues(respMap);
        System.out.println();
        System.out.println();
        System.out.println("timely_response");
        printValues(timelyMap);
        System.out.println();
        System.out.println();
        System.out.println("consumer_disputed?");
        printValues(disputedMap);
        System.out.println();
        System.out.println();
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
