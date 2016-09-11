package chapter04.classification;

import java.io.IOException;
import java.util.Random;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import chapter04.RankedPageData;
import chapter04.cv.Dataset;
import chapter04.cv.Fold;
import chapter04.preprocess.StandardizationPreprocessor;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.SolverType;

public class RocAucExample {

    public static void main(String[] args) throws IOException {
        generatedExample();
        realExample();
    }

    public static void generatedExample() {
        int n = 1000;
        Random rnd = new Random(5);
        double[] score = IntStream.range(0, n).mapToDouble(i -> rnd.nextDouble() * i / n).toArray();

        DoubleStream negative = DoubleStream.generate(() -> 0.0).limit(n / 2);
        DoubleStream positive = DoubleStream.generate(() -> 1.0).limit(n / 2);
        DoubleStream concat = DoubleStream.concat(negative, positive);

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

    public static void realExample() throws IOException {
        Fold split = RankedPageData.readRankedPagesMatrix();

        Dataset train = split.getTrain();
        Dataset test = split.getTest();

        StandardizationPreprocessor preprocessor = StandardizationPreprocessor.train(train);
        train = preprocessor.transform(train);
        test = preprocessor.transform(test);

        LibLinear.mute();

        Parameter param = new Parameter(SolverType.L1R_LR, 0.05, 0.1);
        Model finalModel = LibLinear.train(train, param);

        double[] prediction = LibLinear.predictProba(finalModel, test);
        double[] actual = test.getY();

        RocCurve.plot(actual, prediction);

        double auc = Metrics.auc(actual, prediction);
        System.out.printf("ROC AUC %3.2f%n", auc);
    }
}
