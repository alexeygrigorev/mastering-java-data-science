package chapter05.clustering;

import java.awt.Color;
import java.awt.Dimension;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Random;

import javax.swing.JFrame;
import javax.swing.WindowConstants;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.util.MathUtils;

import com.google.common.base.Stopwatch;
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
import smile.clustering.HierarchicalClustering;
import smile.clustering.linkage.Linkage;
import smile.clustering.linkage.UPGMALinkage;
import smile.data.SparseDataset;
import smile.math.matrix.SingularValueDecomposition;
import smile.plot.Dendrogram;
import smile.plot.PlotCanvas;
import smile.plot.Line.Style;

public class HierarchicalSmile {

    public static void main(String[] args) throws IOException {
        DataFrame<Object> categorical = Categorical.readData();

        Stopwatch stopwatch = Stopwatch.createStarted();
        SparseDataset sparse = SmileOHE.oneHotEncoding(categorical);
        System.out.println("OHE took " + stopwatch.stop());

        stopwatch = Stopwatch.createStarted();
        SingularValueDecomposition svd = SingularValueDecomposition.decompose(sparse.toSparseMatrix(), 30);
        System.out.println("SVD took " + stopwatch.stop());

        stopwatch = Stopwatch.createStarted();
        double[][] data = Projections.project(sparse, svd.getV());
        System.out.println("projection took " + stopwatch.stop());
        stopwatch = Stopwatch.createStarted();
        data = sample(data, 50);
        System.out.println("sample took " + stopwatch.stop());

        stopwatch = Stopwatch.createStarted();
        double[][] proximity = calcualateSquaredEuclidean(data);
        System.out.println("proximity calculation took " + stopwatch.stop());

        stopwatch = Stopwatch.createStarted();
        Linkage linkage = new UPGMALinkage(proximity);
        HierarchicalClustering hc = new HierarchicalClustering(linkage);
        System.out.println("clustering took " + stopwatch.stop());

        showDendrogram(hc);
//        double height = 10.0;
//        int[] labels = hc.partition(height);
    }

    private static void showDendrogram(HierarchicalClustering hc) {
        JFrame frame = new JFrame("Dendrogram");
        frame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);

        PlotCanvas dendrogram = Dendrogram.plot(hc.getTree(), hc.getHeight());
        frame.add(dendrogram);

        frame.setSize(new Dimension(1000, 1000));
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }
    
    private static double[][] sample(double[][] data, int size) {
        return sample(data, size, System.nanoTime());
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

    public static double[][] calcualateCosineDistance(double[][] data) {
        int nrow = data.length;
        double[][] normalized = normalize(data);

        Array2DRowRealMatrix m = new Array2DRowRealMatrix(normalized, false);
        double[][] cosine = m.multiply(m.transpose()).getData();

        for (int i = 0; i < nrow; i++) {
            double[] row = cosine[i];
            for (int j = 0; j < row.length; j++) {
                row[j] = 1 - row[j];
            }
        }

        return cosine;
    }

    private static double[][] normalize(double[][] data) {
        int nrow = data.length;
        double[][] normalized = new double[nrow][];

        for (int i = 0; i < nrow; i++) {
            double[] row = data[i].clone();
            normalized[i] = row;
            double norm = new ArrayRealVector(row, false).getNorm();
            for (int j = 0; j < row.length; j++) {
                row[j] = row[j] / norm;
            }
        }

        return normalized;
    }

    public static double[][] calcualateSquaredEuclidean(double[][] data) {
        int nrow = data.length;

        double[] squared = squareRows(data);

        Array2DRowRealMatrix m = new Array2DRowRealMatrix(data, false);
        double[][] product = m.multiply(m.transpose()).getData();
        double[][] dist = new double[nrow][nrow];

        for (int i = 0; i < nrow; i++) {
            for (int j = i + 1; j < nrow; j++) {
                double d = squared[i] - 2 * product[i][j] + squared[j];
                dist[i][j] = dist[j][i] = d; 
            }
        }

        return dist;
    }

    private static double[] squareRows(double[][] data) {
        int nrow = data.length;

        double[] squared = new double[nrow];
        for (int i = 0; i < nrow; i++) {
            double[] row = data[i];

            double res = 0.0;
            for (int j = 0; j < row.length; j++) {
                res = res + row[j] * row[j];
            }

            squared[i] = res;
        }

        return squared;
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
