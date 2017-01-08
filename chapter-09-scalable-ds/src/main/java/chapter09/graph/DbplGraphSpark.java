package chapter09.graph;

import java.io.File;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.graphframes.GraphFrame;

import com.fasterxml.jackson.jr.ob.JSON;
import com.google.common.collect.Lists;

public class DbplGraphSpark {

    @SuppressWarnings("resource")
    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("graph").setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);

        SparkSession sql = new SparkSession(sc.sc());

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

        Dataset<Row> df = sql.createDataFrame(edges, Edge.class);
        df = df.filter("year >= 1990");
        System.out.println("number of edges: " + df.count());

        df = df.groupBy("node1", "node2").min("year")
               .withColumnRenamed("min(year)", "year")
               .withColumnRenamed("node1", "src")
               .withColumnRenamed("node2", "dst");

        System.out.println("number of edges after grouping: " + df.count());

        df.groupBy("year").count()
            .sort("year").toJavaRDD()
            .collect().forEach(System.out::println);

        Dataset<Row> dfReversed = df
                .withColumnRenamed("src", "dst1")
                .withColumnRenamed("dst", "src")
                .withColumnRenamed("dst1", "dst");

        Dataset<Row> allEdges = df.union(dfReversed);



        Dataset<Row> nodes = df.select("src").withColumnRenamed("src", "id").distinct();

        GraphFrame gf = GraphFrame.apply(nodes, allEdges);

//        GraphFrame pageRank = gf.pageRank().resetProbability(0.1).maxIter(7).run();
//        pageRank.vertices().show();

//        new File("./tmp").mkdir();
//        sc.setCheckpointDir("./tmp");

//        Dataset<Row> connectedComponents = gf.connectedComponents().run();
//        connectedComponents.show();

        // number of connections for each 
//        Dataset<Row> degrees = gf.degrees();
//        degrees.show();

        // too long
        // ArrayList<Object> landmarks = Lists.newArrayList("Alin Deutsch",
        // "Daniela Florescu", "Alon Y. Levy");
        // Dataset<Row> shortestPaths =
        // gf.shortestPaths().landmarks(landmarks).run();
        // shortestPaths.show();

        // Dataset<Row> train = gf.filter("year <= 2013");
        // Dataset<Row> test = df.filter("year >= 2014");

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
