package chapter06.ml;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.commons.io.FileUtils;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;

import com.google.common.base.Stopwatch;

import chapter06.MatrixUtils;
import chapter06.ScoredIndex;
import chapter06.UrlRepository;
import chapter06.text.CountVectorizer;
import chapter06.text.TextUtils;
import chapter06.text.TruncatedSVD;
import smile.data.SparseDataset;

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

        SparseDataset docVectors = vectorizer.fitTransform(texts);
        System.out.println("vectorizing took " + stopwatch.stop());

        stopwatch = Stopwatch.createStarted();
        int n = 150;
        boolean normalize = true;
        TruncatedSVD svd = new TruncatedSVD(n, normalize);
        svd.fit(docVectors);
        double[][] docsLsa = svd.transform(docVectors);
        System.out.println("LSA took " + stopwatch.stop());

        stopwatch = Stopwatch.createStarted();
        List<String> query = TextUtils.tokenize("cheap used cars");
        SparseDataset queryVectors = vectorizer.transfrom(Collections.singletonList(query));
        double[] queryLsa = svd.transform(queryVectors)[0];

        double[] lsaScores = MatrixUtils.vectorSimilarity(docsLsa, queryLsa);
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
