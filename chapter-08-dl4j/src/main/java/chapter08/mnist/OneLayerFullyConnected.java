package chapter08.mnist;

import java.io.IOException;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
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

public class OneLayerFullyConnected {

    private static final Logger LOGGER = LoggerFactory.getLogger(OneLayerFullyConnected.class);

    public static void main(String[] args) throws IOException {
        int batchSize = 128;
        int numEpochs = 10;

        int seed = 1;
        int numrow = 28;
        int numcol = 28;

        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, seed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, seed);

        NeuralNetConfiguration.Builder config = new NeuralNetConfiguration.Builder();
        config.seed(seed);

        config.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        config.learningRate(0.005);
        config.regularization(true).l2(0.0001);

        ListBuilder architecture = config.list();

        DenseLayer.Builder innerLayer = new DenseLayer.Builder();
        innerLayer.nIn(numrow * numcol);
        innerLayer.nOut(1000);
        innerLayer.activation("tanh");
        innerLayer.weightInit(WeightInit.UNIFORM);

        architecture.layer(0, innerLayer.build());

        LossFunction loss = LossFunction.NEGATIVELOGLIKELIHOOD;
        OutputLayer.Builder outputLayer = new OutputLayer.Builder(loss);
        outputLayer.nIn(1000);
        outputLayer.nOut(10);
        outputLayer.activation("softmax");
        outputLayer.weightInit(WeightInit.UNIFORM);

        architecture.layer(1, outputLayer.build());

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

        System.out.println(eval.stats());

    }
}
