package chapter10.search;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.SerializationUtils;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.queryparser.analyzing.AnalyzingQueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;

import chapter07.searchengine.FeatureExtractor;
import chapter07.searchengine.PrepareData.QueryDocumentPair;
import joinery.DataFrame;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.XGBoost;
import smile.classification.SoftClassifier;

public class LuceneRerank {

    public static void main(String[] args) throws Exception {
        String userQuery = "cheap used cars";

        Booster booster = XGBoost.loadModel("project/xgb_model.bin");
        FeatureExtractor fe = load("project/feature-extractor.bin");

        File index = new File("project/lucene-rerank");
        FSDirectory directory = FSDirectory.open(index.toPath());
        DirectoryReader reader = DirectoryReader.open(directory);
        IndexSearcher searcher = new IndexSearcher(reader);

        WhitespaceAnalyzer analyzer = new WhitespaceAnalyzer();
        AnalyzingQueryParser parser = new AnalyzingQueryParser("bodyText", analyzer);

        Query query = parser.parse(userQuery);

        TopDocs result = searcher.search(query, 100);
        List<QueryDocumentPair> data = wrapResultsToObject(userQuery, searcher, result);

        DataFrame<Double> featuresDf = fe.transform(data);
        double[][] matrix = featuresDf.toModelMatrix(0.0);
        double[] probs = XgbUtils.predict(booster, matrix);

        List<ScoredIndex> scored = ScoredIndex.wrap(probs);
        for (ScoredIndex idx : scored) {
            QueryDocumentPair doc = data.get(idx.getIndex());
            System.out.printf("%.4f: %s, %s%n", idx.getScore(), doc.getTitle(), doc.getUrl());
        }
    }

    private static List<QueryDocumentPair> wrapResultsToObject(String userQuery, IndexSearcher searcher, TopDocs result)
            throws IOException {
        List<QueryDocumentPair> data = new ArrayList<>();

        for (ScoreDoc scored : result.scoreDocs) {
            int docId = scored.doc;
            Document doc = searcher.doc(docId);

            String url = doc.get("url");
            String title = doc.get("title");
            String bodyText = doc.get("bodyText");
            String allHeaders = doc.get("allHeaders");
            String h1 = doc.get("h1");
            String h2 = doc.get("h2");
            String h3 = doc.get("h3");

            QueryDocumentPair pair = new QueryDocumentPair(userQuery, url, title, bodyText, allHeaders, h1, h2, h3);
            data.add(pair);
        }

        return data;
    }

    public static double[] predict(SoftClassifier<double[]> model, double[][] X) {
        double[] result = new double[X.length];

        double[] probs = new double[2];
        for (int i = 0; i < X.length; i++) {
            model.predict(X[i], probs);
            result[i] = probs[1];
        }

        return result;
    }

    private static <E> E load(String filepath) throws IOException {
        Path path = Paths.get(filepath);
        try (InputStream is = Files.newInputStream(path)) {
            try (BufferedInputStream bis = new BufferedInputStream(is)) {
                return SerializationUtils.deserialize(bis);
            }
        }
    }
}
