package chapter05.dimred;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.Collectors;

import org.jooq.lambda.tuple.Tuple2;

import com.aol.cyclops.control.LazyReact;
import com.aol.cyclops.types.futurestream.LazyFutureStream;
import com.google.common.base.Stopwatch;
import com.google.common.collect.Maps;
import com.google.common.collect.Ordering;
import com.google.common.collect.Sets;

import chapter05.preprocess.SmileOHE;
import joinery.DataFrame;
import smile.data.Datum;
import smile.data.SparseDataset;
import smile.math.SparseArray;
import smile.math.matrix.SingularValueDecomposition;

public class DimRedDistances {

    public static void main(String[] args) throws IOException {
        DataFrame<Object> categorical = Categorical.readData();

        Stopwatch stopwatch = Stopwatch.createStarted();
        SparseDataset sparse = SmileOHE.oneHotEncoding(categorical);
        System.out.println("OHE took " + stopwatch.stop());

        System.out.println("dimensionality: " + sparse.size() + " x " + sparse.ncols());

        int nrows = sparse.size();
        int[] sampleRows = sample(nrows, 10);

        System.out.println(Arrays.toString(sampleRows));

        ExecutorService executor = ForkJoinPool.commonPool();

        Ordering<Tuple2<Long, Double>> byDistance = Ordering.from((t1, t2) -> t1.v2.compareTo(t2.v2));

        PrintWriter pw = new PrintWriter("distances.txt", "UTF-8");

        for (int row : sampleRows) {
            stopwatch = Stopwatch.createStarted();
            SparseArray r1 = sparse.get(row).x;

            LazyFutureStream<Datum<SparseArray>> stream = LazyReact.parallelBuilder().withExecutor(executor)
                    .from(sparse.iterator());

            List<Tuple2<Long, Double>> dist = stream.zipWithIndex().filter(t -> t.v2.intValue() != row).map(t -> {
                SparseArray array = t.v1.x;
                double d = sparseDist(r1, array);
                return new Tuple2<>(t.v2, d);
            }).sorted(byDistance).toList();

            System.out.println("KNN took " + stopwatch.stop());
            String distances = dist.stream().map(t -> t.v2.toString()).collect(Collectors.joining(" "));
            pw.println(distances);
            pw.flush();
        }

        pw.close();

        stopwatch = Stopwatch.createStarted();
        SingularValueDecomposition svd = SingularValueDecomposition.decompose(sparse.toSparseMatrix(), 100);
        System.out.println("SVD took " + stopwatch.stop());

        stopwatch = Stopwatch.createStarted();
        double[][] proj = Projections.project(sparse, svd.getV());
        System.out.println("projection took " + stopwatch.stop());

        pw = new PrintWriter("distances-svd.txt", "UTF-8");

        for (int row : sampleRows) {
            stopwatch = Stopwatch.createStarted();
            double[] r1 = proj[row];

            LazyFutureStream<double[]> stream = LazyReact.parallelBuilder().withExecutor(executor)
                    .fromStream(Arrays.stream(proj));

            List<Tuple2<Long, Double>> dist = stream.zipWithIndex().filter(t -> t.v2.intValue() != row).map(t -> {
                double[] array = t.v1;
                double d = denseDist(r1, array);
                return new Tuple2<>(t.v2, d);
            }).sorted(byDistance).toList();

            System.out.println("dense KNN took " + stopwatch.stop());
            String distances = dist.stream().map(t -> t.v2.toString()).collect(Collectors.joining(" "));
            pw.println(distances);
            pw.flush();
        }

        pw.close();

        executor.shutdown();
    }

    private static double denseDist(double[] r1, double[] r2) {
        double res = 0;
        for (int i = 0; i < r1.length; i++) {
            double diff = r1[i] - r2[i];
            res = res + diff * diff;
        }

        return Math.sqrt(res);
    }

    private static double sparseDist(SparseArray r1, SparseArray r2) {
        Map<Integer, Double> m1 = asMap(r1);
        Map<Integer, Double> m2 = asMap(r2);

        double res = 0;

        Set<Integer> allKeys = Sets.union(m1.keySet(), m2.keySet());
        for (Integer idx : allKeys) {
            double v1 = m1.getOrDefault(idx, 0.0);
            double v2 = m2.getOrDefault(idx, 0.0);
            double diff = v1 - v2;
            res = res + diff * diff;
        }

        return Math.sqrt(res);
    }

    private static Map<Integer, Double> asMap(SparseArray array) {
        Map<Integer, Double> result = Maps.newHashMap();
        array.forEach(e -> result.put(e.i, e.x));
        return result;
    }

    private static int[] sample(int nrows, int n) {
        Random random = new Random(1);
        return random.ints().map(i -> Math.abs(i % nrows)).limit(n).toArray();
    }

}
