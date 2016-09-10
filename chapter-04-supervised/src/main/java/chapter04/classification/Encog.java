package chapter04.classification;

import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;

import chapter04.cv.Dataset;

public class Encog {

    public static void learningCurves(Dataset train, Dataset test, BasicNetwork network, MLTrain trainer,
            int noEpochs) {
        for (int i = 0; i < noEpochs; i++) {
            trainer.iteration();
            if (i % 10 == 0) {
                double aucTrain = auc(network, train);
                double aucVal = auc(network, test);

                System.out.printf("%3d - train:%.4f, val:%.4f%n", i, aucTrain, aucVal);
            }
        }
    }

    public static double auc(BasicNetwork network, Dataset dataset) {
        double[] predictTrain = predict(network, dataset);
        return Metrics.auc(dataset.getY(), predictTrain);
    }

    public static double[] predict(BasicNetwork model, Dataset dataset) {
        double[][] X = dataset.getX();
        double[] result = new double[X.length];

        for (int i = 0; i < X.length; i++) {
            MLData out = model.compute(new BasicMLData(X[i]));
            result[i] = out.getData(0);
        }

        return result;

    }

    public static BasicMLDataSet asEncogDataset(Dataset train) {
        return new BasicMLDataSet(train.getX(), to2d(train.getY()));
    }

    private static double[][] to2d(double[] y) {
        double[][] res = new double[y.length][];

        for (int i = 0; i < y.length; i++) {
            res[i] = new double[] { y[i] };
        }

        return res;
    }

    public static void shutdown() {
        org.encog.Encog.getInstance().shutdown();
    }
}
