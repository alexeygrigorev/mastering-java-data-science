package chapter07.xgb;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.lang3.SerializationUtils;

import com.google.common.collect.ImmutableMap;

import chapter07.StandardizationPreprocessor;
import chapter07.cv.Dataset;
import chapter07.cv.Split;
import chapter07.text.TruncatedSVD;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.IEvaluation;
import ml.dmlc.xgboost4j.java.IObjective;
import ml.dmlc.xgboost4j.java.XGBoost;
import smile.validation.MSE;

public class PerformancePrediction {

    public static void main(String[] args) throws Exception {
        Path path = Paths.get("data/performance.bin");
        Dataset dataset = read(path);

        StandardizationPreprocessor preprocessor = StandardizationPreprocessor.train(dataset);
        dataset = preprocessor.transform(dataset);

        Split trainTestSplit = dataset.shuffleSplit(0.3);
        Dataset allTrain = trainTestSplit.getTrain();

        Split split = allTrain.trainTestSplit(0.3);

        Dataset train = split.getTrain();
        Dataset val = split.getTest();
        TruncatedSVD svd = new TruncatedSVD(100, false).fit(train);
        train = dimred(train, svd);
        val = dimred(val, svd);

        DMatrix dtrain = XgbUtils.wrapData(train);
        DMatrix dval = XgbUtils.wrapData(val);
        Map<String, DMatrix> watches = ImmutableMap.of("train", dtrain, "val", dval);

        IObjective obj = null;
        IEvaluation eval = null;

        Map<String, Object> params = XgbUtils.defaultParams();
        params.put("objective", "reg:linear");
        params.put("eval_metric", "rmse");
        int nrounds = 100;

        Booster model;
        model = XGBoost.train(dtrain, params, nrounds, watches, obj, eval);

        DMatrix dtrainall = XgbUtils.wrapData(allTrain);
        watches = ImmutableMap.of("trainall", dtrainall);
        nrounds = 50;
        model = XGBoost.train(dtrainall, params, nrounds, watches, obj, eval);

        Dataset test = trainTestSplit.getTest();
        double[] predict = XgbUtils.predict(model, test);
        double testRmse = rmse(test.getY(), predict);
        System.out.printf("test rmse: %.4f%n", testRmse);

        

    }


    private static Dataset dimred(Dataset train, TruncatedSVD svd) {
        double[][] transformedX = svd.transform(train);
        int n = svd.getN();
        List<String> featureNames = IntStream.range(0, n)
                .mapToObj(i -> "svd_feature_" + i)
                .collect(Collectors.toList()); 
        return new Dataset(transformedX, train.getY(), featureNames);
    }


    private static Dataset read(Path path) throws IOException {
        if (!path.toFile().exists()) {
            PerformanceDataPreparation.prepareData();
        }

        try (InputStream is = Files.newInputStream(path)) {
            return SerializationUtils.deserialize(is);
        }
    }

    private static double rmse(double[] truth, double[] prediction) {
        double mse = new MSE().measure(truth, prediction);
        return Math.sqrt(mse);
    }

}
