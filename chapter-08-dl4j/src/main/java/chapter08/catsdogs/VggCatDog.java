package chapter08.catsdogs;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.regex.Pattern;

import org.apache.commons.io.FileUtils;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.split.CollectionInputSplit;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.CombinedPreProcessor;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer.PoolingType;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.nativeblas.Nd4jBlas;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Lists;

import chapter08.Metrics;

public class VggCatDog {

    private static final Logger LOGGER = LoggerFactory.getLogger(VggCatDog.class);

    private static final int seed = 1;
    private static final int height = 128;
    private static final int width = 128;
    private static final int channels = 3;
    private static final int batchSize = 30;
    private static final int numClasses = 2;

    public static void main(String[] args) throws IOException {
        ensureNativeLibsAreLoaded();

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        StatsListener statsListener = new StatsListener(statsStorage);

        File root;
        if (args.length == 0) {
            root = new File("/home/agrigorev/tmp/data/cats-dogs");
        } else {
            root = new File(args[0]);
        }

        List<URI> trainUris = readImages(new File(root, "train_cv"));
        List<URI> trainUrisAug = readImages(new File(root, "train_cv_simple"));
        trainUris.addAll(trainUrisAug);

        List<URI> valUris = readImages(new File(root, "val_cv"));

        CombinedPreProcessor.Builder preprocessorBuilder = new CombinedPreProcessor.Builder();
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        preprocessorBuilder.addPreProcessor(scaler);
        CombinedPreProcessor preprocessor = preprocessorBuilder.build();

        MultiLayerNetwork model = createNetwork();
        ScoreIterationListener scoreListener = new ScoreIterationListener(1);
        model.setListeners(scoreListener, statsListener);

        DataSetIterator valSet = datasetIterator(valUris);
        valSet.setPreProcessor(preprocessor);

        for (int epoch = 0; epoch < 20000; epoch++) {
            ArrayList<URI> uris = new ArrayList<>(trainUris);
            Collections.shuffle(uris);
            List<List<URI>> partitions = Lists.partition(uris, batchSize * 20);

            for (List<URI> set : partitions) {
                DataSetIterator trainSet = datasetIterator(set);
                trainSet.setPreProcessor(preprocessor);

                INDArray oldParams = model.params().dup();

                model.fit(trainSet);

                showChangeInWeights(model, oldParams);
                showTrainPredictions(trainSet, model);
                showLogloss(model, valSet, epoch);
            }

            saveModel(model, epoch);
            LOGGER.info("*** Completed epoch {} ***", epoch);
        }

    }

    private static void saveModel(MultiLayerNetwork model, int epoch) throws IOException {
        File folder = new File("models");
        folder.mkdir();

        File locationToSave = new File(folder, "cats_dogs_" + epoch + ".zip"); 
        boolean saveUpdater = true;
        ModelSerializer.writeModel(model, locationToSave, saveUpdater);
    }

    private static MultiLayerNetwork createNetwork() {
        NeuralNetConfiguration.Builder config = new NeuralNetConfiguration.Builder();
        config.seed(seed);
        config.weightInit(WeightInit.RELU);
        config.activation("relu");

        config.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        config.learningRate(0.001);
        config.updater(Updater.RMSPROP);
        config.rmsDecay(0.99);
        ListBuilder network = config.list();

        int l = 0;

        ConvolutionLayer cnn1 = new ConvolutionLayer.Builder(3, 3).name("cnn1").stride(1, 1).nIn(3).nOut(32).biasInit(0)
                .activation("relu").build();
        network.layer(l++, cnn1);
        ConvolutionLayer cnn2 = new ConvolutionLayer.Builder(3, 3).name("cnn2").stride(1, 1).nIn(32).nOut(32)
                .biasInit(0).activation("relu").build();
        network.layer(l++, cnn2);

        SubsamplingLayer pool1 = new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
                .name("pool1").build();
        network.layer(l++, pool1);

        ConvolutionLayer cnn3 = new ConvolutionLayer.Builder(3, 3).name("cnn3").stride(1, 1).nIn(32).nOut(64)
                .biasInit(0).activation("relu").build();
        network.layer(l++, cnn3);
        ConvolutionLayer cnn4 = new ConvolutionLayer.Builder(3, 3).name("cnn4").stride(1, 1).nIn(64).nOut(64)
                .biasInit(0).activation("relu").build();
        network.layer(l++, cnn4);

        SubsamplingLayer pool2 = new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
                .name("pool2").build();
        network.layer(l++, pool2);

        ConvolutionLayer cnn5 = new ConvolutionLayer.Builder(3, 3).name("cnn5").stride(1, 1).nIn(64).nOut(128)
                .biasInit(0).activation("relu").build();
        network.layer(l++, cnn5);
        ConvolutionLayer cnn6 = new ConvolutionLayer.Builder(3, 3).name("cnn6").stride(1, 1).nIn(128).nOut(128)
                .biasInit(0).activation("relu").build();
        network.layer(l++, cnn6);

        SubsamplingLayer pool3 = new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
                .name("pool3").build();
        network.layer(l++, pool3);

        DenseLayer dense1 = new DenseLayer.Builder().name("ffn1").nOut(512).biasInit(0).dropOut(0.0).build();
        network.layer(l++, dense1);
        DenseLayer dense2 = new DenseLayer.Builder().name("ffn2").nOut(256).biasInit(0).dropOut(0.0).build();
        network.layer(l++, dense2);

        OutputLayer output = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).name("output")
                .nOut(2).activation("softmax").build();
        network.layer(l++, output);

        network.backprop(true).pretrain(false);

        network.setInputType(InputType.convolutionalFlat(height, width, channels));

        MultiLayerNetwork model = new MultiLayerNetwork(network.build());
        return model;
    }

    private static void showTrainPredictions(DataSetIterator trainSet, MultiLayerNetwork model) {
        trainSet.reset();
        DataSet ds = trainSet.next();
        INDArray pred = model.output(ds.getFeatureMatrix(), false).get(NDArrayIndex.all(),
                NDArrayIndex.point(0));
        System.out.println("train pred: " + pred);
    }

    private static void showChangeInWeights(MultiLayerNetwork model, INDArray oldParams) {
        INDArray newParams = model.params();

        INDArray diff = newParams.sub(oldParams);
        double change = diff.muli(diff).sumNumber().doubleValue();
        System.out.println("change in weights: " + change);
    }

    private static void ensureNativeLibsAreLoaded() {
        Nd4jBlas blas = new Nd4jBlas();
        blas.close();
    }

    private static void showLogloss(MultiLayerNetwork model, DataSetIterator data, int epoch) throws AssertionError {
        data.reset();

        List<double[]> actuals = new ArrayList<>();
        List<double[]> preds = new ArrayList<>();

        while (data.hasNext()) {
            DataSet ds = data.next();
            INDArray out = model.output(ds.getFeatureMatrix(), false);

            double[] actual = firstColumn(ds.getLabels());
            actuals.add(actual);
            double[] predicted = firstColumn(out);
            preds.add(predicted);

            if (actual.length != predicted.length) {
                throw new AssertionError(String.format("we have a problem! %d != %d", actual.length, predicted.length));
            }
        }

        data.reset();

        double[] allActual = actuals.stream().flatMapToDouble(d -> Arrays.stream(d)).toArray();
        double[] allPred = preds.stream().flatMapToDouble(d -> Arrays.stream(d)).toArray();

        for (int i = 0; i < 50; i++) {
            System.out.printf("(%.0f, %.2f), ", allActual[i], allPred[i]);
        }
        System.out.println();

        double loss = Metrics.logLoss(allActual, allPred);
        LOGGER.info("step {} logloss: {}", epoch, loss);
    }

    private static double[] firstColumn(INDArray arr) {
        int[] shape = arr.shape();
        if (shape.length > 2 || shape[1] > 2) {
            throw new IllegalArgumentException("cannot convert it to 1d array of doubles");
        }
        int len = shape[0];

        double[] result = new double[len];
        for (int i = 0; i < len; i++) {
            result[i] = arr.getDouble(i, 0);
        }

        return result;
    }

    private static DataSetIterator datasetIterator(List<URI> uris) throws IOException {
        CollectionInputSplit train = new CollectionInputSplit(uris);

        PathLabelGenerator labelMaker = new FileNamePartLabelGenerator();
        ImageRecordReader trainRecordReader = new ImageRecordReader(height, width, channels, labelMaker);
        trainRecordReader.initialize(train);

        RecordReaderDataSetIterator iterator = new RecordReaderDataSetIterator(trainRecordReader, batchSize, 1,
                numClasses);

        return iterator;
    }

    private static List<URI> readImages(File dir) {
        Iterator<File> files = FileUtils.iterateFiles(dir, new String[] { "jpg" }, false);
        List<URI> all = new ArrayList<>();

        while (files.hasNext()) {
            File next = files.next();
            all.add(next.toURI());
        }

        return all;
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

}