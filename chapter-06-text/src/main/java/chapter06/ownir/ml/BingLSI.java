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
import org.apache.commons.math3.linear.ArrayRealVector;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;

import com.google.common.base.Stopwatch;

import chapter06.MatrixUtils;
import chapter06.Projections;
import chapter06.ScoredIndex;
import chapter06.UrlRepository;
import chapter06.text.CountVectorizer;
import chapter06.text.TextUtils;
import smile.data.SparseDataset;
import smile.math.matrix.SingularValueDecomposition;
import smile.math.matrix.SparseMatrix;

public class BingLSI {

    public static void main(String[] args) throws IOException {
        try (UrlRepository urls = new UrlRepository()) {
            run(urls);
        }
    }

    private static void run(UrlRepository urls) throws IOException, FileNotFoundException {
        Stopwatch stopwatch = Stopwatch.createStarted();
        List<List<String>> texts = extractText(urls);
        System.out.println("tokenizing took " + stopwatch.stop());

        stopwatch = Stopwatch.createStarted();

        CountVectorizer vectorizer = CountVectorizer.create()
                .withMinimalDocumentFrequency(10)
                .withIdfTransformation()
                .withL2Normalization()
                .withSublinearTfTransformation()
                .build();

        SparseDataset index = vectorizer.fitTransform(texts);
        System.out.println("vectorizing took " + stopwatch.stop());

        stopwatch = Stopwatch.createStarted();
        List<String> query = TextUtils.tokenize("cheap used cars");

        stopwatch = Stopwatch.createStarted();
        SparseMatrix matrix = index.toSparseMatrix();
        SingularValueDecomposition svd = SingularValueDecomposition.decompose(matrix, 150);
        double[][] lsaIndex = Projections.project(index, svd.getV());
        lsaIndex = MatrixUtils.l2RowNormalize(lsaIndex);


        SparseDataset queryMatrix = vectorizer.transfrom(Arrays.asList(query));
        double[][] queryLsa = Projections.project(queryMatrix, svd.getV());
        queryLsa = l2normalize(queryLsa);
        System.out.println("LSA took " + stopwatch.stop());

        stopwatch = Stopwatch.createStarted();
        double[] lsaScores = MatrixUtils.vectorSimilarity(lsaIndex, queryLsa[0]);
        List<ScoredIndex> scored = ScoredIndex.wrap(lsaScores, 0.2);
        System.out.println("querying took " + stopwatch.stop());

        System.out.println();
        System.out.println("LSA:");

        for (int i = 0; i < 5; i++) {
            ScoredIndex document = scored.get(i);
            List<String> originalTokens = texts.get(document.getIndex());
            System.out.println(document.getScore() + " " + String.join(" ", originalTokens));
        }
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
