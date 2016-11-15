package chapter06.cv;

import java.io.Serializable;
import java.util.Arrays;

public class Dataset implements Serializable {

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

}
