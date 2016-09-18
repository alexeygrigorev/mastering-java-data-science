package chapter04.cv;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import jsat.classifiers.CategoricalData;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.linear.DenseVector;
import jsat.regression.RegressionDataSet;

public class Dataset implements Serializable {

    private static long SEED = 1;

    private final double[][] X;
    private final double[] y;

    public Dataset(double[][] X, double[] y) {
        this.X = X;
        this.y = y;
    }

    public double[][] getX() {
        return X;
    }

    public double[] getY() {
        return y;
    }

    public int[] getYAsInt() {
        return Arrays.stream(y).mapToInt(d -> (int) d).toArray();
    }

    public int length() {
        return getX().length;
    }

    public List<Split> shuffleKFold(int k) {
        return CV.kfold(this, k, true, SEED);
    }

    public List<Split> kfold(int k) {
        return CV.kfold(this, k, false, SEED);
    }

    public Split trainTestSplit(double testRatio) {
        return CV.trainTestSplit(this, testRatio, false, SEED);
    }

    public Split shuffleSplit(double testRatio) {
        return CV.trainTestSplit(this, testRatio, true, SEED);
    }

    public ClassificationDataSet toJsatClassificationDataset() {
        // TODO: what if it's not binary?
        CategoricalData binary = new CategoricalData(2);

        List<DataPointPair<Integer>> data = new ArrayList<>(X.length);
        for (int i = 0; i < X.length; i++) {
            int target = (int) y[i];
            DataPoint row = new DataPoint(new DenseVector(X[i]));
            data.add(new DataPointPair<Integer>(row, target));
        }

        return new ClassificationDataSet(data, binary);
    }

    public RegressionDataSet toJsatRegressionDataset() {
        List<DataPointPair<Double>> data = new ArrayList<>(X.length);

        for (int i = 0; i < X.length; i++) {
            DataPoint row = new DataPoint(new DenseVector(X[i]));
            data.add(new DataPointPair<Double>(row, y[i]));
        }

        return new RegressionDataSet(data);
    }
}
