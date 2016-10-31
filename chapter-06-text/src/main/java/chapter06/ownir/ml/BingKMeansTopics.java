package chapter06.ownir.ml;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.commons.io.FileUtils;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;

import chapter06.CountVectorizer;
import chapter06.Projections;
import chapter06.TextUtils;
import chapter06.UrlRepository;
import chapter06.ownir.ScoredIndex;
import smile.clustering.KMeans;
import smile.data.SparseDataset;
import smile.math.matrix.SingularValueDecomposition;
import smile.math.matrix.SparseMatrix;

public class BingKMeansTopics {

    public static void main(String[] args) throws IOException {
        UrlRepository urls = new UrlRepository();
        List<List<String>> texts = extractText(urls);
        urls.close();

        CountVectorizer vectorizer = CountVectorizer.create()
                .withMinimalDocumentFrequency(10)
                .withIdfTransformation()
                .withL2Normalization()
                .withSublinearTfTransformation()
                .build();

        SparseDataset index = vectorizer.fitTransform(texts);

        SparseMatrix matrix = index.toSparseMatrix();
        SingularValueDecomposition svd = SingularValueDecomposition.decompose(matrix, 150);
        double[][] lsa = Projections.project(index, svd.getV());
        lsa = l2normalize(lsa);

        int maxIter = 100;
        int runs = 3;
        int k = 100;
        KMeans km = new KMeans(lsa, k, maxIter, runs);

        System.out.println(Arrays.toString(km.getClusterSize()));

        double[][] centroids = km.centroids();
        double[][] centroidsOriginal = Projections.project(centroids, t(svd.getV()));

        List<String> featureNames = vectorizer.featureNames();

        for (int centroidId = 0; centroidId < k; centroidId++) {
            System.out.print("cluster no " + centroidId + ": ");
            double[] centroid = centroidsOriginal[centroidId];
            List<ScoredIndex> scored = ScoredIndex.wrap(centroid);
            for (int i = 0; i < 20; i++) {
                ScoredIndex scoredTerm = scored.get(i);
                int position = scoredTerm.getIndex();
                String term = featureNames.get(position);
                System.out.print(term + ", ");
            }

            System.out.println();
        }
     }


    public static double[][] t(double[][] M) {
        Array2DRowRealMatrix matrix = new Array2DRowRealMatrix(M, false);
        return matrix.transpose().getData();
    }


    private static double[][] l2normalize(double[][] data) {
        for (int i = 0; i < data.length; i++) {
            double[] row = data[i];
            ArrayRealVector vector = new ArrayRealVector(row, false);
            double norm = vector.getNorm();
            if (norm != 0) {
                vector.mapDivideToSelf(norm);
                data[i] = vector.getDataRef();
            }
        }

        return data;
    }

    private static List<List<String>> extractText(UrlRepository urls) throws IOException, FileNotFoundException {
        Path path = Paths.get("data/bing-search-results.txt");
        List<String> lines = FileUtils.readLines(path.toFile(), StandardCharsets.UTF_8);

        Stream<List<String>> documents = lines.parallelStream().flatMap(line -> {
            String[] split = line.split("\t");
            String url = split[3];

            Optional<String> html = urls.get(url);
            if (!html.isPresent()) {
                return Stream.empty();
            }

            Document document = Jsoup.parse(html.get());
            Element body = document.body();
            if (body == null) {
                return Stream.empty();
            }

            List<String> tokens = TextUtils.tokenize(body.text());
            tokens = TextUtils.removeStopwords(tokens);
            return Stream.of(tokens);
        });

        return documents.collect(Collectors.toList());
    }
}
