package chapter06.lucene;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Optional;

import org.apache.commons.io.FileUtils;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
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
import me.tongfei.progressbar.ProgressBar;
import me.tongfei.progressbar.ProgressBarStyle;

public class IndexBuilder {

    private static final FieldType URL_FIELD = createUrlFieldType();
    private static final FieldType BODY_FIELD = createBodyFieldType();

    private static final String INDEX_PATH = "index/bing/";

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

        StandardAnalyzer analyzer = new StandardAnalyzer();
        IndexWriter writer = new IndexWriter(directory, new IndexWriterConfig(analyzer));

        Path path = Paths.get("data/bing-search-results.txt");
        List<String> lines = FileUtils.readLines(path.toFile(), StandardCharsets.UTF_8);

        ProgressBar pb = new ProgressBar("lucene index", lines.size(), ProgressBarStyle.ASCII);
        pb.start();

        for (String line : lines) {
            pb.step();

            String[] split = line.split("\t");
            String url = split[3];

            Optional<String> html = urls.get(url);
            if (!html.isPresent()) {
                continue;
            }

            org.jsoup.nodes.Document jsoupDoc = Jsoup.parse(html.get());
            Element body = jsoupDoc.body();
            if (body == null) {
                continue;
            }

            Document doc = new Document();
            doc.add(new Field("url", url, URL_FIELD));
            doc.add(new Field("title", jsoupDoc.title(), URL_FIELD));
            doc.add(new Field("content", body.text(), BODY_FIELD));

            writer.addDocument(doc);
        }

        pb.stop();

        writer.commit();
        writer.close();
        directory.close();

        System.out.println("indexing took " + stopwatch.stop());
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
