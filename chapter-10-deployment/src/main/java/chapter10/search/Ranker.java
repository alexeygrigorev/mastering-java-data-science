package chapter10.search;

import java.util.List;

import chapter07.searchengine.PrepareData.QueryDocumentPair;

public interface Ranker {

    SearchResults rank(List<QueryDocumentPair> inputList) throws Exception;

}
