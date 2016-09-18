package chapter04.classification;

import java.io.IOException;
import java.util.Random;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

public class RocAucExample {

    public static void main(String[] args) throws IOException {
        generatedExample();
    }

    public static void generatedExample() {
        int n = 1000;
        Random rnd = new Random(5);
        double[] score = IntStream.range(0, n).mapToDouble(i -> rnd.nextDouble() * i / n).toArray();

        DoubleStream positive = DoubleStream.generate(() -> 1.0).limit(n / 2);
        DoubleStream negative = DoubleStream.generate(() -> 0.0).limit(n / 2);
        DoubleStream concat = DoubleStream.concat(positive, negative);

        double[] actual = concat.map(d -> maybeFlip(d, 0.1, rnd)).toArray();

        RocCurve.plot(actual, score);
        double auc = Metrics.auc(actual, score);
        System.out.printf("ROC AUC %3.2f%n", auc);
    }

    public static double maybeFlip(double d, double prob, Random rnd) {
        boolean notFlip = rnd.nextDouble() > prob;
        if (notFlip) {
            return d;
        }

        if (d == 1.0) {
            return 0.0;
        }

        return 1.0;
    }
}
