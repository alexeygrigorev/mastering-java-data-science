package chapter09.graph;

import java.io.File;
import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.graphframes.GraphFrame;

import com.fasterxml.jackson.jr.ob.JSON;
import com.google.common.collect.Sets;

import chapter09.Metrics;
import ml.dmlc.xgboost4j.scala.Booster;
import ml.dmlc.xgboost4j.scala.EvalTrait;
import ml.dmlc.xgboost4j.scala.ObjectiveTrait;
import ml.dmlc.xgboost4j.scala.spark.XGBoost;
import ml.dmlc.xgboost4j.scala.spark.XGBoostModel;
import scala.Predef;
import scala.Tuple2;
import scala.collection.JavaConversions;
import scala.collection.immutable.Map;
import scala.collection.mutable.WrappedArray;

public class DblpGraphSpark {

    public static void main(String[] args) {
        SparkConf conf = new SparkConf()
                .setAppName("graph")
                .setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);

        SparkSession sql = new SparkSession(sc.sc());

        Dataset<Row> df = readData(sc, sql);

        Dataset<Row> trainInput = df.filter("year <= 2013");

        Dataset<Row> features = calculateFeatures(sc, sql, trainInput, "train");
        features.show();

        features = features.drop("id", "node1", "node2");
        features = features.withColumn("rnd", functions.rand(1));

        //trainLogReg(features);

        Dataset<Row> trainFeatures = features.filter("rnd <= 0.8").drop("rnd");
        Dataset<Row> valFeatures = features.filter("rnd > 0.8").drop("rnd");

        List<String> columns = Arrays.asList("commonFriends", "totalFriendsApprox", 
                "jaccard", "pagerank_mult", "pagerank_max", "pagerank_min", 
                "pref_attachm", "degree_max", "degree_min", "same_comp");

        JavaRDD<LabeledPoint> trainRdd = trainFeatures.toJavaRDD().map(r -> {
            Vector vec = toDenseVector(columns, r);
            double label = r.getAs("target");
            return new LabeledPoint(label, vec);
        });

        System.out.println("training a model");

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
                XGBoost.train(trainData, params, nRounds, numWorkers, objective, eval, externalMemoryCache, nanValue);

        Booster xgb = model._booster();

        JavaRDD<Pair<float[], Double>> valRdd = valFeatures.toJavaRDD().map(r -> {
            float[] vec = rowToFloatArray(columns, r);
            double label = r.getAs("target");
            return ImmutablePair.of(vec, label);
        });

//        model.eval(JavaRDD.toRDD(valRdd), "auc", null, 20, false);
//        model.eval(JavaRDD.toRDD(valRdd), "logloss", null, 20, false);

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

    private static <K, V> Map<K, V> toScala(HashMap<K, V> params) {
        return JavaConversions.mapAsScalaMap(params)
                .toMap(Predef.<Tuple2<K, V>>conforms());
    }

    private static void trainLogReg(Dataset<Row> features) {
        Dataset<Row> trainFeatures = features.filter("rnd <= 0.8").drop("rnd");
        Dataset<Row> valFeatures = features.filter("rnd > 0.8").drop("rnd");

        List<String> columns = Arrays.asList("commonFriends", "totalFriendsApprox", 
                "jaccard", "pagerank_mult", "pagerank_max", "pagerank_min", 
                "pref_attachm", "degree_max", "degree_min", "same_comp");

        JavaRDD<org.apache.spark.mllib.regression.LabeledPoint> trainRdd = trainFeatures.toJavaRDD().map(r -> {
                org.apache.spark.mllib.linalg.Vector vec = toDenseMllibVector(columns, r);
            double label = r.getAs("target");
            return new org.apache.spark.mllib.regression.LabeledPoint(label, vec);
        });

        LogisticRegressionModel logreg = new LogisticRegressionWithLBFGS()
                    .run(JavaRDD.toRDD(trainRdd));

        System.out.println("predicting...");
        logreg.clearThreshold();

        JavaRDD<Pair<Double, Double>> predRdd = valFeatures.toJavaRDD().map(r -> {
            org.apache.spark.mllib.linalg.Vector v = toDenseMllibVector(columns, r);
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

    private static DenseVector toDenseVector(List<String> columns, Row r) {
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

    private static float[] rowToFloatArray(List<String> columns, Row r) {
        int featureVecLen = columns.size();
        float[] values = new float[featureVecLen];
        for (int i = 0; i < featureVecLen; i++) {
            Object o = r.getAs(columns.get(i));
            values[i] = castToFloat(o);
        }
        return values;
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

    private static org.apache.spark.mllib.linalg.Vector toDenseMllibVector(List<String> columns, Row r) {
        double[] values = rowToDoubleArray(columns, r);
        return new org.apache.spark.mllib.linalg.DenseVector(values);
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

    private static Dataset<Row> readData(JavaSparkContext sc, SparkSession sql) {
        Dataset<Row> df;
        File graphParquet = new File("tmp/graph.parquet");
        if (graphParquet.exists()) {
            System.out.println("reading graph from " + graphParquet);
            return sql.read().parquet(graphParquet.getAbsolutePath());
        } 

        JavaRDD<String> edgeFile = sc.textFile("/home/agrigorev/tmp/data/dblp/dblp_coauthorship.json.gz");
        JavaRDD<Edge> edges = edgeFile.filter(s -> s.length() > 1).map(s -> {
            Object[] array = JSON.std.arrayFrom(s);

            String node1 = (String) array[0];
            String node2 = (String) array[1];
            Integer year = (Integer) array[2];

            if (year == null) {
                return new Edge(node1, node2, -1);
            }

            return new Edge(node1, node2, year);
        });

        edges.take(10).forEach(System.out::println);

        df = sql.createDataFrame(edges, Edge.class);

        df = df.filter("year >= 1990");
        System.out.println("number of edges: " + df.count());

        df = df.groupBy("node1", "node2").min("year")
               .withColumnRenamed("min(year)", "year");

        System.out.println("number of edges after grouping: " + df.count());

        df.groupBy("year").count()
            .sort("year").toJavaRDD()
            .collect().forEach(System.out::println);

        df.write().parquet(graphParquet.getAbsolutePath());
        return df;
    }

    private static Dataset<Row> calculateFeatures(JavaSparkContext sc, SparkSession sql, Dataset<Row> df, String datasetType) {
        File featuresParquet = new File("tmp/all_features_" + datasetType + ".parquet");

        if (featuresParquet.exists()) {
            System.out.println("reading all features from " + featuresParquet);
            return sql.read().parquet(featuresParquet.getAbsolutePath());
        }

        Pair<Dataset<Row>, Dataset<Row>> graph = prepareGraph(sql, df, datasetType);
        Dataset<Row> nodes = graph.getLeft();
        nodes.show();

        Dataset<Row> edges = graph.getRight();
        edges.show();

        GraphFrame gf = new GraphFrame(nodes, edges);

        new File("./tmp/ckp").mkdirs();
        sc.setCheckpointDir("./tmp/ckp");

        Dataset<Row> pageRank = pageRank(sql, datasetType, gf);
        pageRank.show();

        Dataset<Row> connectedComponents = connectedComponents(sql, datasetType, gf);
        connectedComponents.show();

        Dataset<Row> degrees = degrees(sql, datasetType, gf);
        degrees.show();

        Dataset<Row> train = trainEdges(sql, df, datasetType, nodes, edges);
        train.show();

        Dataset<Row> commonFriends = calculateCommonFriends(sql, datasetType, edges, train);
        commonFriends.show();

        Dataset<Row> totalFriends = calculateTotalFriends(sql, datasetType, edges, train);
        totalFriends.show();

        Dataset<Row> jaccard = calculateJaccard(sql, edges, train, datasetType);
        jaccard.show();

        Dataset<Row> nodeFeatures = nodeFeatures(sql, pageRank, connectedComponents, degrees, train, datasetType);
        nodeFeatures.show();

         Dataset<Row> join = train.join(commonFriends, "id")
             .join(totalFriends, "id")
             .join(jaccard, "id")
             .join(nodeFeatures, "id");

        join.write().parquet(featuresParquet.getAbsolutePath());
        return join;
    }

    private static Dataset<Row> nodeFeatures(SparkSession sql, Dataset<Row> pageRank, Dataset<Row> connectedComponents,
            Dataset<Row> degrees, Dataset<Row> train, String datasetType) {
        File nodeParquet = new File("tmp/node_features_" + datasetType + ".parquet");

        if (nodeParquet.exists()) {
            System.out.println("reading jaccard from " + nodeParquet);
            return sql.read().parquet(nodeParquet.getAbsolutePath());
        }

        Dataset<Row> nodeFeatures = pageRank.join(degrees, "id").join(connectedComponents, "id");
        nodeFeatures = nodeFeatures.withColumnRenamed("id", "node_id");
        nodeFeatures.show();

        Dataset<Row> join = train.drop("target");

        join = join.join(nodeFeatures, join.col("node1").equalTo(nodeFeatures.col("node_id")));
        join = join.drop("node_id")
                .withColumnRenamed("pagerank", "pagerank_1")
                .withColumnRenamed("degree", "degree_1")
                .withColumnRenamed("component", "component_1");

        join = join.join(nodeFeatures, join.col("node2").equalTo(nodeFeatures.col("node_id")));
        join = join.drop("node_id")
                .withColumnRenamed("pagerank", "pagerank_2")
                .withColumnRenamed("degree", "degree_2")
                .withColumnRenamed("component", "component_2");

        join = join.drop("node1", "node2");

        join = join
            .withColumn("pagerank_mult", join.col("pagerank_1").multiply(join.col("pagerank_2")))
            .withColumn("pagerank_max", functions.greatest("pagerank_1", "pagerank_2"))
            .withColumn("pagerank_min", functions.least("pagerank_1", "pagerank_2"))
            .withColumn("pref_attachm", join.col("degree_1").multiply(join.col("degree_2")))
            .withColumn("degree_max", functions.greatest("degree_1", "degree_2"))
            .withColumn("degree_min", functions.least("degree_1", "degree_2"))
            .withColumn("same_comp", join.col("component_1").equalTo(join.col("component_2")));

        join = join.drop("pagerank_1", "pagerank_2");
        join = join.drop("degree_1", "degree_2");
        join = join.drop("component_1", "component_2");

        join.write().parquet(nodeParquet.getAbsolutePath());
        return join;
    }

    private static Dataset<Row> calculateJaccard(SparkSession sql, Dataset<Row> edges, Dataset<Row> train, String datasetType) {
        File jaccardParquet = new File("tmp/jaccard_" + datasetType + ".parquet");

        if (jaccardParquet.exists()) {
            System.out.println("reading jaccard from " + jaccardParquet);
            return sql.read().parquet(jaccardParquet.getAbsolutePath());
        }

        Dataset<Row> e = edges.drop("node1", "node2", "year", "target");
        Dataset<Row> coAuthors = e.groupBy("src")
                .agg(functions.collect_set("dst").as("others"))
                .withColumnRenamed("src", "node");
        coAuthors.show();

        Dataset<Row> join = train.drop("target");
        join = join.join(coAuthors, join.col("node1").equalTo(coAuthors.col("node")));
        join = join.drop("node").withColumnRenamed("others", "others1");

        join = join.join(coAuthors, join.col("node2").equalTo(coAuthors.col("node")));
        join = join.drop("node").withColumnRenamed("others", "others2");
        join = join.drop("node1", "node2");

        JavaRDD<Row> jaccardRdd = join.toJavaRDD().map(r -> {
            long id = r.getAs("id");
            WrappedArray<Long> others1 = r.getAs("others1");
            WrappedArray<Long> others2 = r.getAs("others2");

            Set<Long> set1 = Sets.newHashSet((Long[]) others1.array());
            Set<Long> set2 = Sets.newHashSet((Long[]) others2.array());

            int intersection = Sets.intersection(set1, set2).size();
            int union = Sets.union(set1, set2).size();

            double jaccard = intersection / (union + 1.0);

            return RowFactory.create(id, jaccard);
        });

        StructField node1Field = DataTypes.createStructField("id", DataTypes.LongType, false);
        StructField node2Field = DataTypes.createStructField("jaccard", DataTypes.DoubleType, false);
        StructType schema = DataTypes.createStructType(Arrays.asList(node1Field, node2Field));

        Dataset<Row> jaccard = sql.createDataFrame(jaccardRdd, schema);
        jaccard.write().parquet(jaccardParquet.getAbsolutePath());

        return jaccard;
    }

    private static Pair<Dataset<Row>, Dataset<Row>> prepareGraph(SparkSession sql, Dataset<Row> df, String datasetType) {
        File nodesParquet = new File("tmp/nodes_" + datasetType +  ".parquet");
        File edgesParquet = new File("tmp/edges_" + datasetType +  ".parquet");
        Dataset<Row> nodes;
        Dataset<Row> edges;

        if (nodesParquet.exists()) {
            System.out.println("reading nodes from " + nodesParquet);
            nodes = sql.read().parquet(nodesParquet.getAbsolutePath());

            System.out.println("reading edges from " + edgesParquet);
            edges = sql.read().parquet(edgesParquet.getAbsolutePath());

            return ImmutablePair.of(nodes, edges);
        }

        Dataset<Row> dfReversed = df
                .withColumnRenamed("node1", "tmp")
                .withColumnRenamed("node2", "node1")
                .withColumnRenamed("tmp", "node2")
                .select("node1", "node2", "year");

        edges = df.union(dfReversed);

        nodes = edges.select("node1").withColumnRenamed("node1", "node").distinct();
        nodes = nodes.withColumn("id", functions.monotonicallyIncreasingId());
        nodes.write().parquet(nodesParquet.getAbsolutePath());

        edges = edges.join(nodes, edges.col("node2").equalTo(nodes.col("node")));
        edges = edges.drop("node").withColumnRenamed("id", "dst");

        edges = edges.join(nodes, edges.col("node1").equalTo(nodes.col("node")));
        edges = edges.drop("node").withColumnRenamed("id", "src");

        edges.write().parquet(edgesParquet.getAbsolutePath());

        return ImmutablePair.of(nodes, edges);
    }

    private static Dataset<Row> calculateTotalFriends(SparkSession sql, String datasetType, Dataset<Row> edges,
            Dataset<Row> train) {
        File tfParquet = new File("tmp/total_friends_" + datasetType + ".parquet");

        if (tfParquet.exists()) {
            System.out.println("reading common friends from " + tfParquet);
            return sql.read().parquet(tfParquet.getAbsolutePath());
        } 

        Dataset<Row> e = edges.drop("node1", "node2", "year", "target");

        Dataset<Row> join = train.join(e, 
                train.col("node1").equalTo(edges.col("src")));

        Dataset<Row> totalFriends = join.select("id", "dst")
                .groupBy("id")
                .agg(functions.approxCountDistinct("dst").as("totalFriendsApprox"));

        totalFriends.write().parquet(tfParquet.getAbsolutePath());
        return totalFriends;
    }

    private static Dataset<Row> calculateCommonFriends(SparkSession sql, String datasetType, Dataset<Row> edges,
            Dataset<Row> train) {
        File cfParquet = new File("tmp/common_friends_" + datasetType + ".parquet");

        if (cfParquet.exists()) {
            System.out.println("reading common friends from " + cfParquet);
            return sql.read().parquet(cfParquet.getAbsolutePath());
        }

        Dataset<Row> commonFriends;
        Dataset<Row> e1 = edges.drop("node1", "node2", "year", "target")
                .withColumnRenamed("src", "e1_src")
                .withColumnRenamed("dst", "e1_dst")
                .as("e1");
        Dataset<Row> e2 = edges.drop("node1", "node2", "year", "target")
                .withColumnRenamed("src", "e2_src")
                .withColumnRenamed("dst", "e2_dst")
                .as("e2");

        Dataset<Row> join = train.join(e1, 
                train.col("node1").equalTo(e1.col("e1_src")));
        join = join.join(e2, 
                join.col("node2").equalTo(e2.col("e2_src")).and(
                join.col("e1_dst").equalTo(e2.col("e2_dst"))));

        commonFriends = join.groupBy("id").count();
        commonFriends = commonFriends.withColumnRenamed("count", "commonFriends");

        commonFriends.write().parquet(cfParquet.getAbsolutePath());
        return commonFriends;
    }

    private static Dataset<Row> trainEdges(SparkSession sql, Dataset<Row> df, String datasetType, Dataset<Row> nodes,
            Dataset<Row> edges) {
        File trainEdgesParquet = new File("tmp/train_edges_" + datasetType + ".parquet");

        if (trainEdgesParquet.exists()) {
            System.out.println("reading train edges from " + trainEdgesParquet);
            return sql.read().parquet(trainEdgesParquet.getAbsolutePath());
        }

        Dataset<Row> trainEdges;
        Dataset<Row> pos = positiveSamples(df, nodes);
        pos.show();

        Dataset<Row> negSimple = simpleNegatives(sql, nodes);
        negSimple.show();

        Dataset<Row> hardNeg = hardNegatives(edges);
        hardNeg.show();

        trainEdges = pos.union(negSimple).union(hardNeg);
        trainEdges = trainEdges.withColumn("id", functions.monotonicallyIncreasingId());

        trainEdges.describe("target").show();
        trainEdges.write().parquet(trainEdgesParquet.getAbsolutePath());

        return trainEdges;
    }

    private static Dataset<Row> degrees(SparkSession sql, String datasetType, GraphFrame gf) {
        File degreeParquet = new File("tmp/degrees_" + datasetType + ".parquet");
        if (degreeParquet.exists()) {
            System.out.println("reading degrees from " + degreeParquet);
            return sql.read().parquet(degreeParquet.getAbsolutePath());
        }

        Dataset<Row> degrees;
        System.out.println("computing degrees");
        degrees = gf.degrees();
        degrees.write().parquet(degreeParquet.getAbsolutePath());
        return degrees;
    }

    private static Dataset<Row> connectedComponents(SparkSession sql, String datasetType, GraphFrame gf) {
        File ccParquet = new File("tmp/connected_components_" + datasetType + ".parquet");

        if (ccParquet.exists()) {
            System.out.println("reading connected components from " + ccParquet);
            return sql.read().parquet(ccParquet.getAbsolutePath());
        }

        Dataset<Row> connectedComponents;
        System.out.println("computing connected components");
        connectedComponents = gf.connectedComponents().run();
        connectedComponents = connectedComponents.drop("node");
        connectedComponents.write().parquet(ccParquet.getAbsolutePath());
        return connectedComponents;
    }

    private static Dataset<Row> pageRank(SparkSession sql, String datasetType, GraphFrame gf) {
        File pageRankParquet = new File("tmp/page_rank_text_" + datasetType +  ".parquet");

        if (pageRankParquet.exists()) {
            System.out.println("reading page rank from " + pageRankParquet);
            return sql.read().parquet(pageRankParquet.getAbsolutePath());
        } 

        Dataset<Row> pageRankVertices;
        System.out.println("computing page rank");
        GraphFrame pageRank = gf.pageRank().resetProbability(0.1).maxIter(7).run();
        pageRankVertices = pageRank.vertices();
        pageRankVertices = pageRankVertices.drop("node");
        pageRankVertices.write().parquet(pageRankParquet.getAbsolutePath());

        return pageRankVertices;
    }

    private static Dataset<Row> hardNegatives(Dataset<Row> edges) {
        Dataset<Row> e1 = edges.drop("node1", "node2", "year")
                .withColumnRenamed("src", "e1_src")
                .withColumnRenamed("dst", "e1_dst")
                .as("e1");
        Dataset<Row> e2 = edges.drop("node1", "node2", "year")
                .withColumnRenamed("src", "e2_src")
                .withColumnRenamed("dst", "e2_dst")
                .as("e2");

        Column diffDest = e1.col("e1_dst").notEqual(e2.col("e2_dst"));
        Column sameSrc = e1.col("e1_src").equalTo(e2.col("e2_src"));
        Dataset<Row> hardNeg = e1.join(e2, diffDest.and(sameSrc));
        hardNeg = hardNeg.select("e1_dst", "e2_dst")
                .withColumnRenamed("e1_dst", "node1")
                .withColumnRenamed("e2_dst", "node2");

        hardNeg = hardNeg.withColumn("rnd", functions.rand(0));
        hardNeg = hardNeg.filter("rnd >= 0.95").drop("rnd");
        hardNeg = hardNeg.limit(6_000_000);
        hardNeg = hardNeg.withColumn("target", functions.lit(0.0));

        return hardNeg;
    }

    private static Dataset<Row> simpleNegatives(SparkSession sql, Dataset<Row> nodes) {
        Dataset<Row> nodeIds = nodes.select("id");

        long nodesCount = nodeIds.count();
        double fraction = 12_000_000.0 / nodesCount;

        Dataset<Row> sample1 = nodeIds.sample(true, fraction, 1);
        sample1 = sample1.withColumn("rnd", functions.rand(1)).orderBy("rnd").drop("rnd");

        Dataset<Row> sample2 = nodeIds.sample(true, fraction, 2);
        sample2 = sample2.withColumn("rnd", functions.rand(2)).orderBy("rnd").drop("rnd");

        long sample1Count = sample1.count();
        long sample2Count = sample2.count();
        int minSize = (int) Math.min(sample1Count, sample2Count);

        sample1 = sample1.limit(minSize);
        sample2 = sample2.limit(minSize);

        JavaRDD<Row> sample1Rdd = sample1.toJavaRDD();
        JavaRDD<Row> sample2Rdd = sample2.toJavaRDD();

        JavaRDD<Row> concat = sample1Rdd.zip(sample2Rdd).map(t -> {
            long id1 = t._1.getLong(0);
            long id2 = t._2.getLong(0);
            return RowFactory.create(id1, id2);
        });

        StructField node1Field = DataTypes.createStructField("node1", DataTypes.LongType, false);
        StructField node2Field = DataTypes.createStructField("node2", DataTypes.LongType, false);
        StructType schema = DataTypes.createStructType(Arrays.asList(node1Field, node2Field));

        Dataset<Row> negSimple = sql.createDataFrame(concat, schema);
        negSimple = negSimple.withColumn("target", functions.lit(0.0));

        return negSimple;
    }

    private static Dataset<Row> positiveSamples(Dataset<Row> df, Dataset<Row> nodes) {
        Dataset<Row> pos = df.drop("year");
        pos = pos.join(nodes, pos.col("node1").equalTo(nodes.col("node")));
        pos = pos.drop("node", "node1").withColumnRenamed("id", "node1");

        pos = pos.join(nodes, pos.col("node2").equalTo(nodes.col("node")));
        pos = pos.drop("node", "node2").withColumnRenamed("id", "node2");

        pos = pos.withColumn("target", functions.lit(1.0));
        return pos;
    }


    public static class Edge implements Serializable {
        private final String node1;
        private final String node2;
        private final int year;

        public Edge(String node1, String node2, int year) {
            this.node1 = node1;
            this.node2 = node2;
            this.year = year;
        }

        public String getNode1() {
            return node1;
        }

        public String getNode2() {
            return node2;
        }

        public int getYear() {
            return year;
        }

        @Override
        public String toString() {
            return "Edge [node1=" + node1 + ", node2=" + node2 + ", year=" + year + "]";
        }

    }
}
