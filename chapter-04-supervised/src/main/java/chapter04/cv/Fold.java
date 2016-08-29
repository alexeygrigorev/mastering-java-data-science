package chapter04.cv;

import java.util.Objects;

public class Fold {

    private final Dataset train;
    private final Dataset test;

    public Fold(Dataset train, Dataset test) {
        this.train = train;
        this.test = test;
    }

    public static Fold fromIndexes(Dataset dataset, int[] trainIndex, int[] testIndex) {
        double[][] X = dataset.getX();
        double[] y = dataset.getY();

        int trainSize = trainIndex.length;

        double[][] trainXres = new double[trainSize][];
        double[] trainYres = new double[trainSize];
        for (int i = 0; i < trainSize; i++) {
            int idx = trainIndex[i];
            trainXres[i] = X[idx];
            trainYres[i] = y[idx];
        }

        int testSize = testIndex.length;

        double[][] testXres = new double[testSize][];
        double[] testYres = new double[testSize];
        for (int i = 0; i < testSize; i++) {
            int idx = testIndex[i];
            testXres[i] = X[idx];
            testYres[i] = y[idx];
        }

        Dataset train = new Dataset(trainXres, trainYres);
        Dataset test = new Dataset(testXres, testYres);
        return new Fold(train, test);
    }

    public Dataset getTrain() {
        return train;
    }

    public Dataset getTest() {
        return test;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof Fold) {
            Fold other = (Fold) obj;
            return train.equals(other.train) && test.equals(test);
        }

        return false;
    }

    @Override
    public int hashCode() {
        return Objects.hash(train, test);
    }
}
