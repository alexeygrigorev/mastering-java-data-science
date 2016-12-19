package chapter08.mnist;

import java.io.IOException;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TwoLayersFullyConnected {

    private static final Logger LOGGER = LoggerFactory.getLogger(TwoLayersFullyConnected.class);

    public static void main(String[] args) throws IOException {
        int batchSize = 128;
        int seed = 1;
        int numEpochs = 20;

        int numrow = 28;
        int numcol = 28;

        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, seed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, seed);

        NeuralNetConfiguration.Builder config = new NeuralNetConfiguration.Builder();
        config.seed(seed);

        config.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        config.iterations(1);

        config.learningRate(0.005);

        config.updater(Updater.SGD);
        config.regularization(true).l2(0.0001);

        ListBuilder architecture = config.list();

        DenseLayer.Builder innerLayer1 = new DenseLayer.Builder();
        innerLayer1.nIn(numrow * numcol);
        innerLayer1.nOut(1000);
        innerLayer1.activation("tanh");
        innerLayer1.dropOut(0.5);
        innerLayer1.weightInit(WeightInit.UNIFORM);

        architecture.layer(0, innerLayer1.build());

        DenseLayer.Builder innerLayer2 = new DenseLayer.Builder();
        innerLayer2.nIn(1000);
        innerLayer2.nOut(2000);
        innerLayer2.activation("tanh");
        innerLayer2.dropOut(0.5);
        innerLayer2.weightInit(WeightInit.UNIFORM);

        architecture.layer(1, innerLayer2.build());

        LossFunction loss = LossFunction.NEGATIVELOGLIKELIHOOD;
        OutputLayer.Builder outputLayer = new OutputLayer.Builder(loss);
        outputLayer.nIn(2000);
        outputLayer.nOut(10);
        outputLayer.activation("softmax");
        outputLayer.weightInit(WeightInit.UNIFORM);

        architecture.layer(2, outputLayer.build());

        architecture.pretrain(false);
        architecture.backprop(true);

        MultiLayerNetwork nn = new MultiLayerNetwork(architecture.build());
        nn.init();

        nn.setListeners(new ScoreIterationListener(1));

        LOGGER.info("training started");

        for (int i = 0; i < numEpochs; i++) {
            nn.fit(mnistTrain);
        }

        LOGGER.info("Evaluate model....");
        Evaluation eval = new Evaluation(10);
        while (mnistTest.hasNext()) {
            DataSet next = mnistTest.next();
            INDArray output = nn.output(next.getFeatures());
            eval.eval(next.getLabels(), output);
        }

        LOGGER.info(eval.stats());

    }
}
