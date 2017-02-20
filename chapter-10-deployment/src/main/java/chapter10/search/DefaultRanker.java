package chapter10.search;

import java.util.List;

import chapter07.searchengine.PrepareData.QueryDocumentPair;

public class DefaultRanker implements Ranker {

    private static final String NAME = "lucene";

    @Override
    public SearchResults rank(List<QueryDocumentPair> inputList) throws Exception {
        return SearchResults.wrap(NAME, inputList);
    }

    @Override
    public String name() {
        return NAME;
    }

}
