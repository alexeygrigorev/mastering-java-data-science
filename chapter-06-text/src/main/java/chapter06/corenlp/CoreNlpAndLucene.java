package chapter06.corenlp;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

import org.apache.commons.io.FileUtils;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.FSDirectory;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Element;

import com.google.common.base.Stopwatch;

import chapter06.UrlRepository;

public class CoreNlpAndLucene {

    private static final FieldType URL_FIELD = createUrlFieldType();
    private static final FieldType BODY_FIELD = createBodyFieldType();

    private static final String INDEX_PATH = "index/bing_nps/";

    public static void main(String[] args) throws Exception {
        try (UrlRepository urls = new UrlRepository()) {
            createIndexIfNeeded(urls);
        }
    }

    private static void createIndexIfNeeded(UrlRepository urls) throws IOException, FileNotFoundException {
        File index = new File(INDEX_PATH);
        if (index.exists()) {
            System.out.println("Index already exists - do nothing");
            return;
        }

        Stopwatch stopwatch = Stopwatch.createStarted();

        index.mkdirs();
        System.out.println("index is not created - creating it...");
        FSDirectory directory = FSDirectory.open(index.toPath());

        Analyzer analyzer = new WhitespaceAnalyzer();
        IndexWriter writer = new IndexWriter(directory, new IndexWriterConfig(analyzer));

        Path path = Paths.get("data/bing-search-results.txt");
        List<String> lines = FileUtils.readLines(path.toFile(), StandardCharsets.UTF_8);

        StanfordNlpTokenizer tokenizer = new StanfordNlpTokenizer();

        lines.parallelStream().forEach(line -> {
            Optional<Document> doc = convertToDocument(urls, tokenizer, line);
            if (doc.isPresent()) {
                try {
                    writer.addDocument(doc.get());
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });


        writer.commit();
        writer.close();
        directory.close();

        System.out.println("indexing took " + stopwatch.stop());
    }

    private static Optional<Document> convertToDocument(UrlRepository urls, StanfordNlpTokenizer tokenizer, String line) {
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

        String titleTokens = tokenize(tokenizer, jsoupDoc.title());
        String bodyTokens = tokenize(tokenizer, body.text());

        Document doc = new Document();
        doc.add(new Field("url", url, URL_FIELD));
        doc.add(new Field("title", titleTokens, URL_FIELD));
        doc.add(new Field("content", bodyTokens, BODY_FIELD));

        return Optional.of(doc);
    }

    

    private static String tokenize(StanfordNlpTokenizer tokenizer, String text) {
        List<Word> tokens = tokenizer.tokenize(text);
        return tokens.stream()
                    .map(Word::getLemma)
                    .map(String::toLowerCase)
                    .collect(Collectors.joining(" "));
    }

    private static FieldType createUrlFieldType() {
        FieldType field = new FieldType();
        field.setTokenized(false);
        field.setStored(true);
        field.freeze();
        return field;
    }

    private static FieldType createBodyFieldType() {
        FieldType field = new FieldType();
        field.setStored(false);
        field.setTokenized(true);
        field.setIndexOptions(IndexOptions.DOCS_AND_FREQS);
        field.freeze();
        return field;
    }

}
