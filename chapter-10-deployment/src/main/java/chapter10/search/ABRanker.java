package chapter10.search;

import java.util.List;
import java.util.Random;

import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Multiset;

import chapter07.searchengine.PrepareData.QueryDocumentPair;

public class ABRanker implements FeedbackRanker {

    private final Ranker aRanker;
    private final Ranker bRanker;
    private final Random random;
    private final Multiset<String> count = ConcurrentHashMultiset.create();

    public ABRanker(Ranker aRanker, Ranker bRanker, long seed) {
        this.aRanker = aRanker;
        this.bRanker = bRanker;
        this.random = new Random(seed);
    }

    @Override
    public SearchResults rank(List<QueryDocumentPair> inputList) throws Exception {
        if (random.nextBoolean()) {
            return aRanker.rank(inputList);
        } else {
            return bRanker.rank(inputList);
        }
    }

    @Override
    public void registerClick(String algorithm, String uuid) {
        count.add(algorithm);
    }

}
