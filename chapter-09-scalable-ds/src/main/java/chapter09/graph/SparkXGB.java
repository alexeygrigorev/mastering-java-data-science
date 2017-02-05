package chapter09.graph;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;

import chapter09.Metrics;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;
import ml.dmlc.xgboost4j.scala.Booster;
import ml.dmlc.xgboost4j.scala.EvalTrait;
import ml.dmlc.xgboost4j.scala.ObjectiveTrait;
import ml.dmlc.xgboost4j.scala.spark.XGBoost;
import ml.dmlc.xgboost4j.scala.spark.XGBoostModel;
import scala.Predef;
import scala.Tuple2;
import scala.collection.JavaConversions;
import scala.collection.immutable.Map;

public class SparkXGB {

    private static final List<String> COLUMNS = Arrays.asList("commonFriends", "totalFriendsApprox", "jaccard",
            "pagerank_mult", "pagerank_max", "pagerank_min", "pref_attachm", "degree_max", "degree_min", "same_comp");

    public static Booster train(JavaSparkContext sc, SparkSession sql, Dataset<Row> features)
            throws XGBoostError {
        File xgbModelFile = new File("link_pred_model_xgb.bin");
        if (xgbModelFile.exists()) {
            System.out.println("the model exists, loading it...");
            ml.dmlc.xgboost4j.java.Booster model = ml.dmlc.xgboost4j.java.XGBoost.loadModel(xgbModelFile.getName());
            return new Booster(model);
        }

        System.out.println("training the xgb model...");
        features = features.drop("id", "node1", "node2");
        features = features.withColumn("rnd", functions.rand(1));

        Dataset<Row> trainFeatures = features.filter("rnd <= 0.8").drop("rnd");
        Dataset<Row> valFeatures = features.filter("rnd > 0.8").drop("rnd");

        JavaRDD<LabeledPoint> trainRdd = toLabeledPoints(trainFeatures);
        trainRdd.cache();

        Map<String, Object> params = xgbParams();

        int nRounds = 20;
        int numWorkers = 4;
        ObjectiveTrait objective = null;

        EvalTrait eval = null;
        boolean externalMemoryCache = false;
        float nanValue = Float.NaN;
        RDD<LabeledPoint> trainData = JavaRDD.toRDD(trainRdd); 

        XGBoostModel model = 
                XGBoost.train(trainData, params, nRounds, numWorkers, 
                        objective, eval, externalMemoryCache, nanValue);

        validate(valFeatures, model._booster());

        System.out.println("training full model");
        JavaRDD<LabeledPoint> fullRdd = toLabeledPoints(features);
        trainData = JavaRDD.toRDD(fullRdd);

        XGBoostModel modelFull = 
                XGBoost.train(trainData, params, nRounds, numWorkers, 
                        objective, eval, externalMemoryCache, nanValue);

        Booster xgbFull = modelFull._booster();
        xgbFull.saveModel(xgbModelFile.getName());

        return xgbFull;
    }

    public static Map<String, Object> xgbParams() {
        HashMap<String, Object> params = new HashMap<String, Object>();
        params.put("eta", 0.3);
        params.put("gamma", 0);
        params.put("max_depth", 6);
        params.put("min_child_weight", 1);
        params.put("max_delta_step", 0);
        params.put("subsample", 1);
        params.put("colsample_bytree", 1);
        params.put("colsample_bylevel", 1);
        params.put("lambda", 1);
        params.put("alpha", 0);
        params.put("tree_method", "approx");
        params.put("objective", "binary:logistic");
        params.put("eval_metric", "logloss");
        params.put("nthread", 1);
        params.put("seed", 42);
        params.put("silent", 1);

        return toScala(params);
    }

    private static JavaRDD<LabeledPoint> toLabeledPoints(Dataset<Row> trainFeatures) {
        return trainFeatures.toJavaRDD().map(r -> {
            Vector vec = toDenseVector(COLUMNS, r);
            double label = r.getAs("target");
            return new LabeledPoint(label, vec);
        });
    }

    private static void validate(Dataset<Row> valFeatures, Booster xgb) throws XGBoostError {
        JavaRDD<ml.dmlc.xgboost4j.LabeledPoint> valRdd = valFeatures.toJavaRDD().map(r -> {
            float[] vec = rowToFloatArray(COLUMNS, r);
            double label = r.getAs("target");
            return ml.dmlc.xgboost4j.LabeledPoint.fromDenseVector((float)label, vec);
        });

        List<ml.dmlc.xgboost4j.LabeledPoint> valPoints = valRdd.collect();
        DMatrix data = new DMatrix(valPoints.iterator(), null);


        float[][] xgbPred = xgb.predict(new ml.dmlc.xgboost4j.scala.DMatrix(data), false, 20);

        double[] actual = floatToDouble(data.getLabel());
        double[] predicted = unwrapToDouble(xgbPred);

        new Random(0).ints(100, 0, actual.length).forEach(i -> {
            System.out.printf("actual %.1f, pred %.4f%n", actual[i], predicted[i]);
        });

        double logLoss = Metrics.logLoss(actual, predicted);
        System.out.printf("log loss: %.4f%n", logLoss);

        double auc = Metrics.auc(actual, predicted);
        System.out.printf("auc: %.4f%n", auc);
    }

    public static JavaRDD<ScoredEdge> predict(Booster xgb, Dataset<Row> features) {
        JavaRDD<ScoredEdge> scoredRdd = features.toJavaRDD().mapPartitions(rows -> {
            List<ScoredEdge> scoredEdges = new ArrayList<>();
            List<ml.dmlc.xgboost4j.LabeledPoint> labeled = new ArrayList<>();
            while (rows.hasNext()) {
                Row r = rows.next();

                long node1 = r.getAs("node1");
                long node2 = r.getAs("node2");
                double target = r.getAs("target");

                scoredEdges.add(new ScoredEdge(node1, node2, target));

                float[] vec = rowToFloatArray(COLUMNS, r);
                labeled.add(ml.dmlc.xgboost4j.LabeledPoint.fromDenseVector(0.0f, vec));
            }

            DMatrix data = new DMatrix(labeled.iterator(), null);
            float[][] xgbPred = xgb.predict(new ml.dmlc.xgboost4j.scala.DMatrix(data), false, 20);

            for (int i = 0; i < scoredEdges.size(); i++) {
                double pred = xgbPred[i][0];
                ScoredEdge edge = scoredEdges.get(i);
                edge.setScore(pred);
            }

            return scoredEdges.iterator();
        });

        return scoredRdd;
    }

    private static DenseVector toDenseVector(List<String> columns, Row r) {
        double[] values = rowToDoubleArray(columns, r);
        return new DenseVector(values);
    }

    private static double[] floatToDouble(float[] floats) {
        double[] result = new double[floats.length];
        for (int i = 0; i < floats.length; i++) {
            result[i] = floats[i];
        }
        return result;

    }

    public static double[] unwrapToDouble(float[][] floatResults) {
        int n = floatResults.length;
        double[] result = new double[n];
        for (int i = 0; i < n; i++) {
            result[i] = floatResults[i][0];
        }
        return result;
    }


    private static double[] rowToDoubleArray(List<String> columns, Row r) {
        int featureVecLen = columns.size();
        double[] values = new double[featureVecLen];
        for (int i = 0; i < featureVecLen; i++) {
            Object o = r.getAs(columns.get(i));
            values[i] = castToDouble(o);
        }
        return values;
    }

    private static float[] rowToFloatArray(List<String> columns, Row r) {
        int featureVecLen = columns.size();
        float[] values = new float[featureVecLen];
        for (int i = 0; i < featureVecLen; i++) {
            Object o = r.getAs(columns.get(i));
            values[i] = castToFloat(o);
        }
        return values;
    }

    private static double castToDouble(Object o) {
        if (o instanceof Number) {
            Number number = (Number) o;
            return number.doubleValue();
        }

        if (o instanceof Boolean) {
            Boolean bool = (Boolean) o;
            if (bool) {
                return 1.0;
            } else {
                return 0.0;
            }
        }

        throw new IllegalArgumentException("cannot cast " + o.getClass() + " to double");
    }

    private static float castToFloat(Object o) {
        if (o instanceof Number) {
            Number number = (Number) o;
            return number.floatValue();
        }

        if (o instanceof Boolean) {
            Boolean bool = (Boolean) o;
            if (bool) {
                return 1.0f;
            } else {
                return 0.0f;
            }
        }

        throw new IllegalArgumentException("cannot cast " + o.getClass() + " to float");
    }

    private static <K, V> Map<K, V> toScala(HashMap<K, V> params) {
        return JavaConversions.mapAsScalaMap(params)
                .toMap(Predef.<Tuple2<K, V>>conforms());
    }
}
