package chapter10.search;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.SerializationUtils;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.queryparser.analyzing.AnalyzingQueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;

import chapter07.searchengine.FeatureExtractor;
import chapter07.searchengine.PrepareData.QueryDocumentPair;

public class SearchEngineService {

    public static void main(String[] args) throws Exception {
        FeatureExtractor fe = load("project/feature-extractor.bin");
        XgbRanker ranker = new XgbRanker(fe, "project/xgb_model.bin");

        File index = new File("project/lucene-rerank");
        FSDirectory directory = FSDirectory.open(index.toPath());
        DirectoryReader reader = DirectoryReader.open(directory);
        IndexSearcher searcher = new IndexSearcher(reader);

        SearchEngineService service = new SearchEngineService(searcher, ranker);
        List<SearchResult> search = service.search("cheap used cars");

        search.forEach(System.out::println);
    }

    private final IndexSearcher searcher;
    private final Ranker ranker;

    private final WhitespaceAnalyzer analyzer = new WhitespaceAnalyzer();
    private final AnalyzingQueryParser parser = new AnalyzingQueryParser("bodyText", analyzer);

    public SearchEngineService(IndexSearcher searcher, Ranker ranker) {
        this.searcher = searcher;
        this.ranker = ranker;
    }

    public List<SearchResult> search(String userQuery) throws Exception {
        Query query = parser.parse(userQuery);

        TopDocs result = searcher.search(query, 100);
        List<QueryDocumentPair> data = wrapResultsToObject(userQuery, searcher, result);

        List<QueryDocumentPair> ranked = ranker.rank(data);

        List<SearchResult> searchResult = new ArrayList<>(100);
        for (QueryDocumentPair pair : ranked) {
            String title = pair.getTitle();
            String url = pair.getUrl();
            searchResult.add(new SearchResult(url, title));
        }

        return searchResult;
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

            QueryDocumentPair pair = new QueryDocumentPair(userQuery, url, title, bodyText, allHeaders, h1, h2, h3);
            data.add(pair);
        }

        return data;
    }

    private static <E> E load(String filepath) throws IOException {
        Path path = Paths.get(filepath);
        try (InputStream is = Files.newInputStream(path)) {
            try (BufferedInputStream bis = new BufferedInputStream(is)) {
                return SerializationUtils.deserialize(bis);
            }
        }
    }
}
