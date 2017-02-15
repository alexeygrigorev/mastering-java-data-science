package chapter10.search;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;

import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Multiset;

import chapter07.searchengine.PrepareData.QueryDocumentPair;

public class BanditRanker implements FeedbackRanker {

    private static final int WARM_UP_ROUNDS = 10000;

    private final Map<String, Ranker> rankers;
    private final List<String> rankerNames;
    private final double epsilon;

    private final Multiset<String> counts = ConcurrentHashMultiset.create();
    private final AtomicLong count = new AtomicLong();
    private final Random random;

    public BanditRanker(Map<String, Ranker> rankers, double epsilon, long seed) {
        this.rankers = rankers;
        this.rankerNames = new ArrayList<>(rankers.keySet());
        this.epsilon = epsilon;
        this.random = new Random(seed);
    }

    @Override
    public SearchResults rank(List<QueryDocumentPair> inputList) throws Exception {
        if (count.getAndIncrement() < WARM_UP_ROUNDS) {
            return rankByRandomRanker(inputList);
        }

        double rnd = random.nextDouble();
        if (rnd > epsilon) {
            return rankByBestRanker(inputList);
        }

        return rankByRandomRanker(inputList);
    }

    private SearchResults rankByBestRanker(List<QueryDocumentPair> inputList) throws Exception {
        String rankerName = bestRanker();
        Ranker ranker = rankers.get(rankerName);
        return ranker.rank(inputList);
    }

    private String bestRanker() {
        Comparator<Multiset.Entry<String>> cnp = 
                (e1, e2) -> Integer.compare(e1.getCount(), e2.getCount());
        Multiset.Entry<String> entry = counts.entrySet().stream().max(cnp).get();
        return entry.getElement();
    }

    private SearchResults rankByRandomRanker(List<QueryDocumentPair> inputList) throws Exception {
        int idx = random.nextInt(rankerNames.size());
        String rankerName = rankerNames.get(idx);
        Ranker ranker = rankers.get(rankerName);
        return ranker.rank(inputList);
    }

    @Override
    public void registerClick(String algorithm, String uuid) {
        counts.add(algorithm);
    }

}
