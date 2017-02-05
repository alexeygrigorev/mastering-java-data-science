package chapter09.graph;

import java.io.File;
import java.util.Arrays;
import java.util.Set;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import com.google.common.collect.Sets;

import scala.collection.mutable.WrappedArray;

public class EdgeFeatures {

    public static Dataset<Row> jaccard(SparkSession sql, String datasetType, Dataset<Row> edges, Dataset<Row> train) {
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



    public static Dataset<Row> totalFriends(SparkSession sql, String datasetType, Dataset<Row> edges,
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

    public static Dataset<Row> commonFriends(SparkSession sql, String datasetType, Dataset<Row> edges,
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

}
