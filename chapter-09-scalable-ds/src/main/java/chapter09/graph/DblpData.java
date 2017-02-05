package chapter09.graph;

import java.io.File;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import com.fasterxml.jackson.jr.ob.JSON;

public class DblpData {

    public static Dataset<Row> read(JavaSparkContext sc, SparkSession sql) {

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

        Dataset<Row> df = sql.createDataFrame(edges, Edge.class);

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

}
