package chapter08.cv;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

public class Dataset implements Serializable {

    private static long SEED = 1;

    private final double[][] X;
    private final double[] y;
    private final List<String> featureNames;

    public Dataset(double[][] X, double[] y, List<String> featureNames) {
        this.X = X;
        this.y = y;
        this.featureNames = featureNames;
    }

    public double[][] getX() {
        return X;
    }

    public double[] getY() {
        return y;
    }

    public List<String> getFeatureNames() {
        return featureNames;
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

}
