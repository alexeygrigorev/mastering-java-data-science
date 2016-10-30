package chapter06.lucene;

import java.io.File;
import java.io.IOException;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.queryparser.analyzing.AnalyzingQueryParser;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;

public class IndexQuery {

    private static final String INDEX_PATH = "index/bing/";

    public static void main(String[] args) throws Exception {
        queryIndex("cheap used cars");
    }

    private static void queryIndex(String userQuery) throws IOException, ParseException {
        File index = new File(INDEX_PATH);
        FSDirectory directory = FSDirectory.open(index.toPath());
        DirectoryReader reader = DirectoryReader.open(directory);
        IndexSearcher searcher = new IndexSearcher(reader);

        StandardAnalyzer analyzer = new StandardAnalyzer();
        AnalyzingQueryParser parser = new AnalyzingQueryParser("content", analyzer);

        Query query = parser.parse(userQuery);

        TopDocs result = searcher.search(query, 10);
        ScoreDoc[] scoreDocs = result.scoreDocs;

        for (ScoreDoc scored : scoreDocs) {
            int docId = scored.doc;
            float luceneScore = scored.score;
            Document doc = searcher.doc(docId);
            System.out.println(luceneScore + " " + doc.get("url") + " " + doc.get("title"));
        }
    }

}
