package chapter10.search;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.queryparser.analyzing.AnalyzingQueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;

import chapter07.searchengine.PrepareData.QueryDocumentPair;

public class SearchEngineService {

    private final IndexSearcher searcher;
    private final FeedbackRanker ranker;

    private final WhitespaceAnalyzer analyzer = new WhitespaceAnalyzer();
    private final AnalyzingQueryParser parser = new AnalyzingQueryParser("bodyText", analyzer);

    public SearchEngineService(IndexSearcher searcher, FeedbackRanker ranker) {
        this.searcher = searcher;
        this.ranker = ranker;
    }

    public SearchResults search(String userQuery) throws Exception {
        Query query = parser.parse(userQuery);

        TopDocs result = searcher.search(query, 100);
        List<QueryDocumentPair> data = wrapResultsToObject(userQuery, searcher, result);

        return ranker.rank(data);
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

            data.add(new QueryDocumentPair(userQuery, url, title, bodyText, allHeaders, h1, h2, h3));
        }

        return data;
    }

    public void registerClick(String algorithm, String uuid) {
        ranker.registerClick(algorithm, uuid);
    }

}
