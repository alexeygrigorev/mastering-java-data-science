package chapter10.search;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.stat.inference.TTest;

import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Multiset;

import chapter07.searchengine.PrepareData.QueryDocumentPair;

public class ABRanker implements FeedbackRanker {

    private final Ranker aRanker;
    private final Ranker bRanker;
    private final Random random;

    private final List<String> aResults = new ArrayList<>();
    private final List<String> bResults = new ArrayList<>();
    private final Multiset<String> clicksCount = ConcurrentHashMultiset.create();

    public ABRanker(Ranker aRanker, Ranker bRanker, long seed) {
        this.aRanker = aRanker;
        this.bRanker = bRanker;
        this.random = new Random(seed);
    }

    @Override
    public SearchResults rank(List<QueryDocumentPair> inputList) throws Exception {
        if (random.nextBoolean()) {
            SearchResults results = aRanker.rank(inputList);
            aResults.add(results.getUuid());
            return results;
        } else {
            SearchResults results = bRanker.rank(inputList);
            bResults.add(results.getUuid());
            return results;
        }
    }

    @Override
    public void registerClick(String algorithm, String uuid) {
        clicksCount.add(uuid);
    }

    public void tTest() {
        double[] sampleA = aResults.stream().mapToDouble(u -> clicksCount.count(u)).toArray();
        double[] sampleB = bResults.stream().mapToDouble(u -> clicksCount.count(u)).toArray();

        TTest tTest = new TTest();
        double p = tTest.tTest(sampleA, sampleB);

        System.out.printf("P(sample means are same) = %.3f%n", p);
    }

    @Override
    public String name() {
        throw new UnsupportedOperationException();
    }
}
