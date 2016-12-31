package chapter08.nd4j;

import java.util.Random;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.impl.NormalDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class ND4JTest {

    public static void main(String[] args) {
        INDArray ones = Nd4j.ones(5, 10);
        System.out.println(ones);

        INDArray zeros = Nd4j.zeros(5, 10);
        System.out.println(zeros);

        Random rnd = new Random(10);
        double[] doubles = rnd.doubles(100).toArray();

        INDArray arr1d = Nd4j.create(doubles);
        System.out.println(arr1d);

        INDArray arr2d = Nd4j.create(doubles, new int[] { 10, 10 });
        System.out.println(arr2d);

        INDArray reshaped = arr1d.reshape(10, 10);
        System.out.println(reshaped);

        System.out.println(reshaped.reshape(1, -1));

        doubles = rnd.doubles(3 * 5 * 5).toArray();
        INDArray arr3d = Nd4j.create(doubles, new int[] { 3, 5, 5 });
        System.out.println(arr3d);

        double[][] doubles2d = new double[3][];
        doubles2d[0] = rnd.doubles(5).toArray();
        doubles2d[1] = rnd.doubles(5).toArray();
        doubles2d[2] = rnd.doubles(5).toArray();

        INDArray arr2d2 = Nd4j.create(doubles2d);
        System.out.println(arr2d2);

        int seed = 0;
        INDArray rand = Nd4j.rand(new int[] { 5, 5 }, seed);
        System.out.println(rand);

        INDArray arr3d2 = Nd4j.rand(new int[] { 3, 5, 5 }, new NormalDistribution(0.5, 0.2));
        System.out.println(arr3d2);

        double[] picArray = rnd.doubles(3 * 2 * 5).map(d -> Math.round(d * 255)).toArray();
        INDArray pic = Nd4j.create(picArray).reshape(3, 2, 5);
        System.out.println(pic);

        for (int i = 0; i < 3; i++) {
            INDArray channel = pic.get(NDArrayIndex.point(i));
            System.out.println("channel " + i);
            System.out.println(channel);
            System.out.println();
        }

        INDArray slice = pic.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.interval(2, 4));
        System.out.println(slice);

        INDArray mmul = arr2d.mmul(arr2d);
        System.out.println(mmul);
    }
}
