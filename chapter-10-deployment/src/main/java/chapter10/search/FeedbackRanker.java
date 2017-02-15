package chapter10.search;

public interface FeedbackRanker extends Ranker {

    void registerClick(String algorithm, String uuid);
    
}
