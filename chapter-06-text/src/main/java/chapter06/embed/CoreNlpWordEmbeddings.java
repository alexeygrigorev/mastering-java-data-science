package chapter06.embed;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.SerializationUtils;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Element;

import com.google.common.base.Stopwatch;

import chapter06.ScoredToken;
import chapter06.UrlRepository;
import chapter06.embed.WordEmbeddings.DimRedMethod;
import chapter06.html.JsoupTextExtractor;
import chapter06.text.Document;

public class CoreNlpWordEmbeddings {

    public static void main(String[] args) throws Exception {
        run();
    }

    private static void run() throws Exception {
        List<Document> documents = tokenizeOrLoad();

        int minDf = 30;
        int window = 3;
        double smoothing = 0.5;

        Stopwatch stopwatch = Stopwatch.createStarted();
        PmiCoOccurrenceMatrix coocMatrix = PmiCoOccurrenceMatrix.fit(documents, minDf, window, smoothing);
        System.out.println("building co-occurrence matrix took " + stopwatch.stop());

        List<String> samples = Arrays.asList("cat", "laptop", "germany", "adidas", "limit");

        WordEmbeddings embeddings = WordEmbeddings.createFromCoOccurrence(coocMatrix, 150, DimRedMethod.SVD);

        for (String sample : samples) {
            System.out.println("most similar for " + sample);

            List<ScoredToken> mostSimilar = embeddings.mostSimilar(sample, 10, 0.2);
            for (ScoredToken similar : mostSimilar) {
                System.out.printf(" - %.3f %s %n", similar.getScore(), similar.getToken());
            }

            System.out.println();
        }

        embeddings.save("bing-embed.bin");
    }

    private static List<Document> tokenizeOrLoad() throws Exception {
        Path path = Paths.get("data/bing-tokens.bin");

        if (path.toFile().exists()) {
            try (InputStream is = Files.newInputStream(path)) {
                System.out.println("loading from cache...");
                Stopwatch stopwatch = Stopwatch.createStarted();
                List<Document> results = SerializationUtils.deserialize(is);
                System.out.println("loading cached version took " + stopwatch.stop());
                return results;
            }
        }

        try (UrlRepository urls = new UrlRepository()) {
            Stopwatch stopwatch = Stopwatch.createStarted();
            List<Document> tokenizeUrls = tokenizeUrls(urls);
            System.out.println("parsing and tokenizing took " + stopwatch.stop());

            try (OutputStream os = Files.newOutputStream(path)) {
                SerializationUtils.serialize((Serializable) tokenizeUrls, os);
            }

            return tokenizeUrls;
        }
    }

    private static List<Document> tokenizeUrls(UrlRepository urls) throws IOException, FileNotFoundException {
        Path path = Paths.get("data/bing-search-results.txt");
        List<String> lines = FileUtils.readLines(path.toFile(), StandardCharsets.UTF_8);

        StanfordNlpSentenceTokenizer tokenizer = new StanfordNlpSentenceTokenizer();

        return lines.parallelStream()
                .map(line -> convertToDocument(urls, tokenizer, line))
                .filter(Optional::isPresent)
                .map(Optional::get).collect(Collectors.toList());
    }

    private static Optional<Document> convertToDocument(UrlRepository urls, StanfordNlpSentenceTokenizer tokenizer,
            String line) {
        String[] split = line.split("\t");
        String url = split[3];

        Optional<String> html = urls.get(url);
        if (!html.isPresent()) {
            return Optional.empty();
        }

        org.jsoup.nodes.Document jsoupDoc = Jsoup.parse(html.get());
        Element body = jsoupDoc.body();
        if (body == null) {
            return Optional.empty();
        }

        JsoupTextExtractor textExtractor = new JsoupTextExtractor();
        body.traverse(textExtractor);
        String text = textExtractor.getText();

        Document doc = tokenizer.tokenize(text);
        return Optional.of(doc);
    }

}
