package chapter06.ownir;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.commons.io.FileUtils;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;

import com.google.common.base.Stopwatch;

import chapter06.CountVectorizer;
import chapter06.MatrixUtils;
import chapter06.TextUtils;
import chapter06.UrlRepository;
import smile.data.SparseDataset;
import smile.math.SparseArray;

public class BingPageIndexer {

    public static final boolean USE_N_GRAMS = true;
    public static final int N = 3;

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
        SparseArray queryVector = vectorizer.transfromVector(query);

        double[] scores = MatrixUtils.vectorSimilarity(index, queryVector);
        List<ScoredIndex> scored = ScoredIndex.wrap(scores, 0.02);
        System.out.println("querying took " + stopwatch.stop());

        System.out.println();
        System.out.println("Usual bag of Words:");

        for (int i = 0; i < 5; i++) {
            ScoredIndex document = scored.get(i);
            List<String> originalTokens = texts.get(document.getIndex());
            System.out.println(String.join(" ", originalTokens));
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
            if (USE_N_GRAMS) {
                tokens = TextUtils.ngrams(tokens, 1, N);
            }

            return Stream.of(tokens);
        });

        return documents.collect(Collectors.toList());
    }
}
