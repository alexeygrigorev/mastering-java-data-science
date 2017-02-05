package chapter09.graph;

import java.io.File;
import java.util.Arrays;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class Sample {

    public static Dataset<Row> trainEdges(SparkSession sql, Dataset<Row> df, String datasetType, Dataset<Row> nodes,
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

    public static Dataset<Row> positiveSamples(Dataset<Row> df, Dataset<Row> nodes) {
        Dataset<Row> pos = df.drop("year");
        pos = pos.join(nodes, pos.col("node1").equalTo(nodes.col("node")));
        pos = pos.drop("node", "node1").withColumnRenamed("id", "node1");

        pos = pos.join(nodes, pos.col("node2").equalTo(nodes.col("node")));
        pos = pos.drop("node", "node2").withColumnRenamed("id", "node2");

        pos = pos.withColumn("target", functions.lit(1.0));
        return pos;
    }

    public static Dataset<Row> simpleNegatives(SparkSession sql, Dataset<Row> nodes) {
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

    public static Dataset<Row> hardNegatives(Dataset<Row> edges) {
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

    public static Dataset<Row> selectTestEdges(SparkSession sql, Dataset<Row> fullTest, Dataset<Row> nodes) {
        File parquet = new File("tmp/val_selected_edges.parquet");
        if (parquet.exists()) {
            System.out.println("reading selected edges from " + parquet);
            return sql.read().parquet(parquet.getAbsolutePath());
        } 

        Dataset<Row> testNodes = fullTest.sample(true, 0.05, 1).select("node1").dropDuplicates();
        Dataset<Row> testEdges = fullTest.join(testNodes, "node1");

        Dataset<Row> join = testEdges.drop("year");

        join = join.join(nodes, join.col("node1").equalTo(nodes.col("node")));
        join = join.drop("node", "node1").withColumnRenamed("id", "node1");

        join = join.join(nodes, join.col("node2").equalTo(nodes.col("node")));
        join = join.drop("node", "node2").withColumnRenamed("id", "node2");

        System.out.println(join.count());
        join.write().parquet(parquet.getAbsolutePath());
        return join;
    }

    public static Dataset<Row> createTestCandidates(SparkSession sql, Dataset<Row> edges, Dataset<Row> selected) {
        File parquet = new File("tmp/val_candidates.parquet");
        if (parquet.exists()) {
            System.out.println("reading val candidates from " + parquet);
            return sql.read().parquet(parquet.getAbsolutePath());
        } 

        Dataset<Row> e1 = selected.select("node1").dropDuplicates();

        Dataset<Row> e2 = edges.drop("node1", "node2", "year")
                .withColumnRenamed("src", "e2_src")
                .withColumnRenamed("dst", "e2_dst")
                .as("e2");

        Column diffDest = e1.col("node1").notEqual(e2.col("e2_dst"));
        Column sameSrc = e1.col("node1").equalTo(e2.col("e2_src"));
        Dataset<Row> candidates = e1.join(e2, diffDest.and(sameSrc));
        candidates = candidates.select("node1", "e2_dst").withColumnRenamed("e2_dst", "node2");

        candidates = candidates.withColumn("target", functions.lit(0.0));

        selected = selected.withColumn("target", functions.lit(1.0));

        candidates = selected.union(candidates).dropDuplicates("node1", "node2");

        candidates = candidates.withColumn("id", functions.monotonicallyIncreasingId());
        candidates.write().parquet(parquet.getAbsolutePath());

        return candidates;
    }

}
