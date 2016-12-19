package chapter08.cv;

import java.util.ArrayList;
import java.util.List;

import smile.data.Datum;
import smile.data.SparseDataset;
import smile.math.SparseArray;
import smile.math.SparseArray.Entry;

public class IndexSplit {
    private final int[] trainIdx;
    private final int[] testIdx;

    public IndexSplit(int[] trainIdx, int[] testIdx) {
        this.trainIdx = trainIdx;
        this.testIdx = testIdx;
    }

    public int[] getTestIdx() {
        return testIdx;
    }

    public int[] getTrainIdx() {
        return trainIdx;
    }

    public static double[] elementsByIndex(double[] array, int[] idx) {
        double[] result = new double[array.length];

        for (int i = 0; i < idx.length; i++) {
            result[i] = array[idx[i]];
        }

        return result;
    }

    public static double[][] elementsByIndex(double[][] array, int[] idx) {
        double[][] result = new double[array.length][];

        for (int i = 0; i < idx.length; i++) {
            result[i] = array[idx[i]];
        }

        return result;
    }

    public static <E> List<E> elementsByIndex(List<E> list, int[] idx) {
        List<E> result = new ArrayList<>(idx.length);

        for (int i = 0; i < idx.length; i++) {
            E el = list.get(idx[i]);
            result.add(el);
        }

        return result;
    }

    public static SparseDataset elementsByIndex(SparseDataset dataset, int[] idx) {
        SparseDataset part = new SparseDataset(dataset.ncols());

        for (int i = 0; i < idx.length; i++) {
            Datum<SparseArray> datum = dataset.get(idx[i]);
            for (Entry e : datum.x) {
                part.set(i, e.i, e.x);
            }
        }

        return part;
    }
}
