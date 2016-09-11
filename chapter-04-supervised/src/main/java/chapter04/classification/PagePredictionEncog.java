package chapter04.classification;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.neural.networks.training.strategy.RegularizationStrategy;

import chapter04.RankedPageData;
import chapter04.cv.Dataset;
import chapter04.cv.Split;
import chapter04.preprocess.StandardizationPreprocessor;

public class PagePredictionEncog {

    public static void main(String[] args) throws Exception {
        Split split = RankedPageData.readRankedPagesMatrix();

        Dataset fullTrain = split.getTrain();
        Dataset test = split.getTest();

        StandardizationPreprocessor preprocessor = StandardizationPreprocessor.train(fullTrain);
        fullTrain = preprocessor.transform(fullTrain);
        test = preprocessor.transform(test);

        Split validationSplit = fullTrain.trainTestSplit(0.3);
        Dataset train = validationSplit.getTrain();

        int noInputNeurons = fullTrain.getX()[0].length;

        BasicNetwork network = new BasicNetwork();
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, noInputNeurons));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 30));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 1));
        network.getStructure().finalizeStructure();
        network.reset(1);

        MLDataSet trainSet = Encog.asEncogDataset(train);

        System.out.println("training  model with many iterations");
        MLTrain trainer = new ResilientPropagation(network, trainSet);
        double lambda = 0.01;
        trainer.addStrategy(new RegularizationStrategy(lambda));

        int noEpochs = 101;
        Encog.learningCurves(validationSplit.getTrain(), validationSplit.getTest(), network, trainer, noEpochs);

        System.out.println();
        System.out.println("retraining full model with 20 iterations");

        network.reset(1);

        MLDataSet fullTrainSet = Encog.asEncogDataset(fullTrain);
        trainer = new ResilientPropagation(network, fullTrainSet);
        trainer.addStrategy(new RegularizationStrategy(lambda));

        Encog.learningCurves(fullTrain, test, network, trainer, 21);

        Encog.shutdown();
    }

}
