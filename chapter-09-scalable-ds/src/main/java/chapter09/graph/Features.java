package chapter09.graph;

import java.io.File;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class Features {

    public static Dataset<Row> prepareTrainFeatures(JavaSparkContext sc, SparkSession sql, Dataset<Row> df, String datasetType) {
        File featuresParquet = new File("tmp/all_features_" + datasetType + ".parquet");

        if (featuresParquet.exists()) {
            System.out.println("reading all features from " + featuresParquet);
            return sql.read().parquet(featuresParquet.getAbsolutePath());
        }

        Pair<Dataset<Row>, Dataset<Row>> graph = GF.prepareGraph(sql, df, datasetType);
        Dataset<Row> nodes = graph.getLeft();
        Dataset<Row> edges = graph.getRight();

        Dataset<Row> train = Sample.trainEdges(sql, df, datasetType, nodes, edges);

        Dataset<Row> nodeFeatures = GF.calculateNodeFeatures(sc, sql, nodes, edges, train, datasetType);

        Dataset<Row> commonFriends = EdgeFeatures.commonFriends(sql, datasetType, edges, train);
        Dataset<Row> totalFriends = EdgeFeatures.totalFriends(sql, datasetType, edges, train);
        Dataset<Row> jaccard = EdgeFeatures.jaccard(sql, datasetType, edges, train);

        Dataset<Row> join = train.join(commonFriends, "id")
             .join(totalFriends, "id")
             .join(jaccard, "id")
             .join(nodeFeatures, "id");

        join.write().parquet(featuresParquet.getAbsolutePath());
        return join;
    }

    public static Dataset<Row> prepareTestFeatures(SparkSession sql, Dataset<Row> edges, Dataset<Row> candidates) {
        File parquet = new File("tmp/val_features.parquet");
        if (parquet.exists()) {
            System.out.println("reading val features from " + parquet);
            return sql.read().parquet(parquet.getAbsolutePath());
        } 

        Dataset<Row> commonFriends = EdgeFeatures.commonFriends(sql, "val", edges, candidates);
        Dataset<Row> totalFriends = EdgeFeatures.totalFriends(sql, "val", edges, candidates);
        Dataset<Row> jaccard = EdgeFeatures.jaccard(sql, "val", edges, candidates);

        Dataset<Row> pageRank = GF.readPageRank(sql, "train");
        Dataset<Row> connectedComponents = GF.readConnectedComponents(sql, "train");
        Dataset<Row> degrees = GF.readDegrees(sql, "train");

        Dataset<Row> nodeFeatures = GF.nodeFeatures(sql, pageRank, connectedComponents, degrees, candidates, "val");

        Dataset<Row> features = candidates.join(commonFriends, "id")
                .join(totalFriends, "id")
                .join(jaccard, "id")
                .join(nodeFeatures, "id");

        features.show();
        features.write().parquet(parquet.getAbsolutePath());
        return features;
    }

}
