package chapter08.mnist;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MnistLeNet {

    private static final Logger LOGGER = LoggerFactory.getLogger(MnistLeNet.class);

    public static void main(String[] args) throws Exception {
        int nChannels = 1;
        int height = 28;
        int width = 28;
        int outputNum = 10;
        int batchSize = 64;
        int nEpochs = 10;
        int iterations = 1;
        int seed = 123;

        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, seed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, seed);

        NeuralNetConfiguration.Builder config = new NeuralNetConfiguration.Builder();
        config.seed(seed);
        config.iterations(iterations);
        config.regularization(true).l2(0.0005);
        config.learningRate(0.01);
        config.weightInit(WeightInit.XAVIER);
        config.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        config.updater(Updater.NESTEROVS).momentum(0.9);
        ListBuilder architect = config.list();

        ConvolutionLayer cnn1 = new ConvolutionLayer.Builder(5, 5)
                .name("cnn1")
                .nIn(nChannels)
                .stride(1, 1)
                .nOut(20)
                .activation("identity")
                .build();
        architect.layer(0, cnn1);

        SubsamplingLayer pool1 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .name("pool1")
                .kernelSize(2, 2)
                .stride(2, 2)
                .build();
        architect.layer(1, pool1);

        ConvolutionLayer cnn2 = new ConvolutionLayer.Builder(5, 5)
                .name("cnn2")
                .stride(1, 1)
                .nOut(50)
                .activation("identity")
                .build();
        architect.layer(2, cnn2);

        SubsamplingLayer pool2 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .name("pool2")
                .kernelSize(2, 2)
                .stride(2, 2)
                .build();
        architect.layer(3, pool2);

        DenseLayer dense1 = new DenseLayer.Builder()
                .name("dense1")
                .activation("relu")
                .nOut(500)
                .build();
        architect.layer(4, dense1);

        OutputLayer output = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .name("output")
                .nOut(outputNum)
                .activation("softmax")
                .build();
        architect.layer(5, output);

        architect.setInputType(InputType.convolutionalFlat(height, width, nChannels));
        architect.backprop(true).pretrain(false);

        MultiLayerNetwork model = new MultiLayerNetwork(architect.build());
        model.init();

        LOGGER.info("Train model....");
        ScoreIterationListener scoreListener = new ScoreIterationListener(1);
        model.setListeners(scoreListener);

        for (int i = 0; i < nEpochs; i++) {
            model.fit(mnistTrain);
            LOGGER.info("*** Completed epoch {} ***", i);

            LOGGER.info("Evaluate model....");
            Evaluation eval = new Evaluation(outputNum);
            while (mnistTest.hasNext()) {
                DataSet ds = mnistTest.next();
                INDArray out = model.output(ds.getFeatureMatrix(), false);
                eval.eval(ds.getLabels(), out);
            }
            LOGGER.info(eval.stats());
            mnistTest.reset();
        }

        ModelSerializer.writeModel(model, "models/le-net.zip", true);

    }

}