package chapter10.search;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

import chapter07.searchengine.PrepareData.QueryDocumentPair;

public class SearchResults {

    private String uuid = UUID.randomUUID().toString();
    private String generatedBy = "na";
    private List<SearchResult> list;

    public static SearchResults wrap(String algorithm, List<QueryDocumentPair> inputList) {
        SearchResults results = new SearchResults();
        results.setGeneratedBy(algorithm);
        results.setList(convert(inputList));
        return results;
    }

    private static List<SearchResult> convert(List<QueryDocumentPair> inputList) {
        List<SearchResult> searchResult = new ArrayList<>(inputList.size());

        for (QueryDocumentPair pair : inputList) {
            String title = pair.getTitle();
            String url = pair.getUrl();
            searchResult.add(new SearchResult(url, title));
        }

        return searchResult;
    }

    public String getUuid() {
        return uuid;
    }

    public void setUuid(String uuid) {
        this.uuid = uuid;
    }

    public String getGeneratedBy() {
        return generatedBy;
    }

    public void setGeneratedBy(String generatedBy) {
        this.generatedBy = generatedBy;
    }

    public List<SearchResult> getList() {
        return list;
    }

    public void setList(List<SearchResult> list) {
        this.list = list;
    }

}
