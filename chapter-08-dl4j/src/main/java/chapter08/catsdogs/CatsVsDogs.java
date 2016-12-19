package chapter08.catsdogs;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.regex.Pattern;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.split.CollectionInputSplit;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CatsVsDogs {
    private static final Logger LOGGER = LoggerFactory.getLogger(CatsVsDogs.class);

    public static void main(String[] args) throws IOException {
        int seed = 1;
        int height = 64;
        int width = 64;
        int channels = 3;
        int batchSize = 10;
        int numClasses = 2;

        String path = args[0];
        File trainDir = new File(path); 
//        File trainDir = new File("/home/agrigorev/tmp/data/cats-dogs/train");
        Pair<List<URI>, List<URI>> data = trainValSplit(trainDir, 0.2, seed);

        List<URI> trainUris = data.getLeft();
        DataSetIterator trainIterator = datasetIterator(height, width, channels, batchSize, numClasses, trainUris);


        List<URI> valUris = data.getRight();
        DataSetIterator valIterator = datasetIterator(height, width, channels, batchSize, numClasses, valUris);


        NeuralNetConfiguration.Builder config = new NeuralNetConfiguration.Builder();
        config.seed(seed);
//        config.iterations(iterations);
        config.regularization(true).l2(0.0005);
        config.learningRate(0.01);
        config.weightInit(WeightInit.XAVIER);
        config.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        config.updater(Updater.NESTEROVS).momentum(0.9);
        config.dropOut(0.2);
        ListBuilder architect = config.list();

        int l = 0;
        architect.layer(l++, new BatchNormalization.Builder().nIn(3).nOut(3).build());

        ConvolutionLayer cnn1 = new ConvolutionLayer.Builder(5, 5).stride(1, 1).padding(0, 0)
                .nIn(3)
                .nOut(32).activation("relu").build();
        architect.layer(l++, cnn1);
        ConvolutionLayer cnn2 = new ConvolutionLayer.Builder(5, 5).stride(1, 1).padding(0, 0)
                .nIn(32)
                .nOut(32).activation("relu").build();
        architect.layer(l++, cnn2);

        ConvolutionLayer cnn3 = new ConvolutionLayer.Builder(5, 5).stride(1, 1).padding(0, 0)
                .nIn(32)
                .nOut(64).activation("relu").build();
        architect.layer(l++, cnn3);
        ConvolutionLayer cnn4 = new ConvolutionLayer.Builder(5, 5).stride(1, 1).padding(0, 0)
                .nIn(64)
                .nOut(64).activation("relu").build();
        architect.layer(l++, cnn4);

        ConvolutionLayer cnn5 = new ConvolutionLayer.Builder(5, 5).stride(1, 1).padding(0, 0)
                .nIn(64)
                .nOut(128).activation("relu").build();
        architect.layer(l++, cnn5);
        ConvolutionLayer cnn6 = new ConvolutionLayer.Builder(5, 5).stride(1, 1).padding(0, 0)
                .nIn(128)
                .nOut(128).activation("relu").build();
        architect.layer(l++, cnn6);

//        ConvolutionLayer cnn7 = new ConvolutionLayer.Builder(5, 5).stride(1, 1).padding(0, 0)
//                .nIn(128)
//                .nOut(256).activation("relu").build();
//        architect.layer(l++, cnn7);
//        ConvolutionLayer cnn8 = new ConvolutionLayer.Builder(5, 5).stride(1, 1).padding(0, 0)
//                .nIn(256)
//                .nOut(256).activation("relu").build();
//        architect.layer(l++, cnn8);

        architect.layer(l++, new DenseLayer.Builder().nOut(500).activation("relu").build());
        architect.layer(l++, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(2)
                .activation("softmax").build());

        architect.setInputType(InputType.convolutionalFlat(64, 64, 1));
        architect.backprop(true).pretrain(false);

        MultiLayerNetwork model = new MultiLayerNetwork(architect.build());
        model.init();

        LOGGER.info("Train model....");

        ScoreIterationListener scoreListener = new ScoreIterationListener(5);
        model.setListeners(scoreListener);

        for (int i = 0; i < 20; i++) {
            model.fit(trainIterator);
            LOGGER.info("*** Completed epoch {} ***", i);

            LOGGER.info("Evaluate model....");
            Evaluation eval = new Evaluation(1);
            while (valIterator.hasNext()) {
                DataSet ds = valIterator.next();
                INDArray out = model.output(ds.getFeatureMatrix(), false);
                eval.eval(ds.getLabels(), out);
            }

            LOGGER.info(eval.stats());
            valIterator.reset();
        }

    }

    private static DataSetIterator datasetIterator(int height, int width, int channels, int batchSize, int numClasses,
            List<URI> uris) throws IOException {
        CollectionInputSplit train = new CollectionInputSplit(uris);

        PathLabelGenerator labelMaker = new FileNamePartLabelGenerator();
        ImageRecordReader trainRecordReader = new ImageRecordReader(height, width, channels, labelMaker);
        trainRecordReader.initialize(train);

        return new RecordReaderDataSetIterator(trainRecordReader, batchSize, 1, numClasses);
    }

    private static Pair<List<URI>, List<URI>> trainValSplit(File trainDir, double testFrac, long seed) {
        Iterator<File> files = FileUtils.iterateFiles(trainDir, new String[] { "jpg" }, false);
        List<URI> all = new ArrayList<>();

        while (files.hasNext()) {
            File next = files.next();
            all.add(next.toURI());
        }

        Random random = new Random(seed);
        Collections.shuffle(all, random);

        int trainSize = (int) (all.size() * (1 - testFrac));
        List<URI> train = all.subList(0, trainSize);
        List<URI> test = all.subList(trainSize, all.size());

        return Pair.of(train, test);
    }

    private static class FileNamePartLabelGenerator implements PathLabelGenerator {
        private static final long serialVersionUID = 1L;

        @Override
        public Writable getLabelForPath(String path) {
            File file = new File(path);
            String name = file.getName();
            String[] split = name.split(Pattern.quote("."));
            return new Text(split[0]);
        }

        @Override
        public Writable getLabelForPath(URI uri) {
            return getLabelForPath(new File(uri).toString());
        }

    }


    private static ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }

    private static ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(new int[]{5,5}, stride, pad).name(name).nOut(out).biasInit(bias).build();
    }

    private static SubsamplingLayer maxPool(String name,  int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[]{2,2}).name(name).build();
    }

}