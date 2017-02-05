package chapter09.graph;

import java.util.List;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import com.google.common.collect.Ordering;

import ml.dmlc.xgboost4j.scala.Booster;

public class LinkPredictionSpark {

    public static void main(String[] args) throws Exception {
        SparkConf conf = new SparkConf().setAppName("graph").setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);

        SparkSession sql = new SparkSession(sc.sc());

        Dataset<Row> df = DblpData.read(sc, sql);
        Dataset<Row> trainInput = df.filter("year <= 2013");
        Dataset<Row> testInput = df.filter("year >= 2014");

        Dataset<Row> trainFeatures = Features.prepareTrainFeatures(sc, sql, trainInput, "train");
        Booster xgb = SparkXGB.train(sc, sql, trainFeatures);

        testModel(sql, testInput, xgb);
    }

    private static void testModel(SparkSession sql, Dataset<Row> testInput, Booster xgb) {
        Pair<Dataset<Row>, Dataset<Row>> graph = GF.readGraph(sql, "train");
        Dataset<Row> nodes = graph.getLeft();
        Dataset<Row> edges = graph.getRight();

        Dataset<Row> selected = Sample.selectTestEdges(sql, testInput, nodes);
        Dataset<Row> candidates = Sample.createTestCandidates(sql, edges, selected);
        Dataset<Row> features = Features.prepareTestFeatures(sql, edges, candidates);

        JavaRDD<ScoredEdge> scoredRdd = SparkXGB.predict(xgb, features);

        JavaPairRDD<Long, List<ScoredEdge>> topSuggestions = scoredRdd
                .keyBy(s -> s.getNode1())
                .groupByKey()
                .mapValues(es -> takeFirst10(es));

        topSuggestions.take(10).forEach(System.out::println);

        double mp10 = topSuggestions.mapToDouble(es -> {
            List<ScoredEdge> es2 = es._2();
            double correct = es2.stream().filter(e -> e.getTarget() == 1.0).count();
            return correct / es2.size();
        }).mean();

        System.out.println(topSuggestions.count());
        System.out.println(mp10);
    }

    private static List<ScoredEdge> takeFirst10(Iterable<ScoredEdge> es) {
        Ordering<ScoredEdge> byScore = Ordering.natural().onResultOf(ScoredEdge::getScore).reverse();
        return byScore.leastOf(es, 10);
    }

}
