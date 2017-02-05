package chapter09.graph;

import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import chapter09.Metrics;

public class SparkMLlib {

    public static void trainLogReg(Dataset<Row> features) {
        Dataset<Row> trainFeatures = features.filter("rnd <= 0.8").drop("rnd");
        Dataset<Row> valFeatures = features.filter("rnd > 0.8").drop("rnd");

        List<String> columns = Arrays.asList("commonFriends", "totalFriendsApprox", 
                "jaccard", "pagerank_mult", "pagerank_max", "pagerank_min", 
                "pref_attachm", "degree_max", "degree_min", "same_comp");

        JavaRDD<LabeledPoint> trainRdd = trainFeatures.toJavaRDD().map(r -> {
            Vector vec = toDenseMllibVector(columns, r);
            double label = r.getAs("target");
            return new org.apache.spark.mllib.regression.LabeledPoint(label, vec);
        });

        LogisticRegressionModel logreg = new LogisticRegressionWithLBFGS()
                    .run(JavaRDD.toRDD(trainRdd));

        System.out.println("predicting...");
        logreg.clearThreshold();

        JavaRDD<Pair<Double, Double>> predRdd = valFeatures.toJavaRDD().map(r -> {
            Vector v = toDenseMllibVector(columns, r);
            double label = r.getAs("target");
            double predict = logreg.predict(v);
            return ImmutablePair.of(label, predict);
        });

        List<Pair<Double, Double>> pred = predRdd.collect();
        pred.subList(0, 10).forEach(System.out::println);

        double[] actual = pred.stream().mapToDouble(Pair::getLeft).toArray();
        double[] predicted = pred.stream().mapToDouble(Pair::getRight).toArray();

        double logLoss = Metrics.logLoss(actual, predicted);
        System.out.printf("log loss: %.4f%n", logLoss);

        double auc = Metrics.auc(actual, predicted);
        System.out.printf("auc: %.4f%n", auc);
    }

    private static Vector toDenseMllibVector(List<String> columns, Row r) {
        double[] values = rowToDoubleArray(columns, r);
        return new DenseVector(values);
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

}
