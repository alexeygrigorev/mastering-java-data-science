package chapter06.project;

import java.io.File;
import java.io.IOException;
import java.util.Map;

import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.FSDirectory;

import com.google.common.collect.ArrayListMultimap;

import chapter06.project.PrepareData.HtmlDocument;

public class CreateLuceneIndex {
    private static final FieldType URL_FIELD = createUrlFieldType();
    private static final FieldType TEXT_FIELD = createTextFieldType();

    public static void main(String[] args) throws IOException {
        File index = new File("project/lucene-rerank");
        index.mkdirs();

        FSDirectory directory = FSDirectory.open(index.toPath());

        WhitespaceAnalyzer analyzer = new WhitespaceAnalyzer();
        IndexWriter writer = new IndexWriter(directory, new IndexWriterConfig(analyzer));

        Map<String, HtmlDocument> docs = PrepareData.readDocumentsFromFile(new File("project/html-cache.bin"));

        for (HtmlDocument htmlDoc : docs.values()) {
            String url = htmlDoc.getUrl();
            String title = htmlDoc.getTitle();
            String bodyText = htmlDoc.getBodyText();
            ArrayListMultimap<String, String> headers = htmlDoc.getHeaders();
            String allHeaders = String.join(" ", headers.values());
            String h1 = String.join(" ", headers.get("h1"));
            String h2 = String.join(" ", headers.get("h2"));
            String h3 = String.join(" ", headers.get("h3"));

            Document doc = new Document();
            doc.add(new Field("url", url, URL_FIELD));
            doc.add(new Field("title", title, TEXT_FIELD));
            doc.add(new Field("bodyText", bodyText, TEXT_FIELD));
            doc.add(new Field("allHeaders", allHeaders, TEXT_FIELD));
            doc.add(new Field("h1", h1, TEXT_FIELD));
            doc.add(new Field("h2", h2, TEXT_FIELD));
            doc.add(new Field("h3", h3, TEXT_FIELD));

            writer.addDocument(doc);
        }

        writer.commit();
        writer.close();
        directory.close();
    }

    private static FieldType createUrlFieldType() {
        FieldType field = new FieldType();
        field.setTokenized(false);
        field.setStored(true);
        field.freeze();
        return field;
    }

    private static FieldType createTextFieldType() {
        FieldType field = new FieldType();
        field.setStored(true);
        field.setTokenized(true);
        field.setIndexOptions(IndexOptions.DOCS_AND_FREQS);
        field.freeze();
        return field;
    }

}
