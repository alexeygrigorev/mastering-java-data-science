package chapter10.search;

import java.util.List;

import chapter07.searchengine.PrepareData.QueryDocumentPair;

public class DefaultRanker implements Ranker {

    @Override
    public SearchResults rank(List<QueryDocumentPair> inputList) throws Exception {
        return SearchResults.wrap("default", inputList);
    }

}
