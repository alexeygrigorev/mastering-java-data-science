package chapter09.graph;

import java.io.File;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.graphframes.GraphFrame;

public class GF {

    public static Pair<Dataset<Row>, Dataset<Row>> prepareGraph(SparkSession sql, Dataset<Row> df, String datasetType) {
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

    public static Pair<Dataset<Row>, Dataset<Row>> readGraph(SparkSession sql, String datasetType) {
        File nodesParquet = new File("tmp/nodes_" + datasetType + ".parquet");
        File edgesParquet = new File("tmp/edges_" + datasetType + ".parquet");

        System.out.println("reading nodes from " + nodesParquet);
        Dataset<Row> nodes = sql.read().parquet(nodesParquet.getAbsolutePath());

        System.out.println("reading edges from " + edgesParquet);
        Dataset<Row> edges = sql.read().parquet(edgesParquet.getAbsolutePath());

        return ImmutablePair.of(nodes, edges);
    }

    public static Dataset<Row> calculateNodeFeatures(JavaSparkContext sc, SparkSession sql, Dataset<Row> nodes, Dataset<Row> edges, Dataset<Row> train, String datasetType) {
        GraphFrame gf = new GraphFrame(nodes, edges);

        new File("./tmp/ckp").mkdirs();
        sc.setCheckpointDir("./tmp/ckp");

        Dataset<Row> pageRank = pageRank(sql, datasetType, gf);
        pageRank.show();

        Dataset<Row> connectedComponents = connectedComponents(sql, datasetType, gf);
        connectedComponents.show();

        Dataset<Row> degrees = degrees(sql, datasetType, gf);
        degrees.show();

        Dataset<Row> nodeFeatures = nodeFeatures(sql, pageRank, connectedComponents, degrees, train, datasetType);
        nodeFeatures.show();

        return nodeFeatures;
    }


    public static Dataset<Row> degrees(SparkSession sql, String datasetType, GraphFrame gf) {
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

    public static Dataset<Row> readDegrees(SparkSession sql, String datasetType) {
        File degreeParquet = new File("tmp/degrees_" + datasetType + ".parquet");
        System.out.println("reading degrees from " + degreeParquet);
        return sql.read().parquet(degreeParquet.getAbsolutePath());
    }

    public static Dataset<Row> connectedComponents(SparkSession sql, String datasetType, GraphFrame gf) {
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

    public static Dataset<Row> readConnectedComponents(SparkSession sql, String datasetType) {
        File ccParquet = new File("tmp/connected_components_" + datasetType + ".parquet");
        System.out.println("reading connected components from " + ccParquet);
        return sql.read().parquet(ccParquet.getAbsolutePath());
    }

    public static Dataset<Row> pageRank(SparkSession sql, String datasetType, GraphFrame gf) {
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

    public static Dataset<Row> readPageRank(SparkSession sql, String datasetType) {
        File pageRankParquet = new File("tmp/page_rank_text_" + datasetType +  ".parquet");
        System.out.println("reading page rank from " + pageRankParquet);
        return sql.read().parquet(pageRankParquet.getAbsolutePath());
    }


    public static Dataset<Row> nodeFeatures(SparkSession sql, Dataset<Row> pageRank, Dataset<Row> connectedComponents,
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

}
