package chapter06;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ScoredIndex implements Comparable<ScoredIndex> {
    private final int indx;
    private final double score;

    public ScoredIndex(int indx, double score) {
        this.indx = indx;
        this.score = score;
    }

    @Override
    public String toString() {
        return String.format("(%s, %.4f)", indx, score);
    }

    public int getIndex() {
        return indx;
    }

    public double getScore() {
        return score;
    }

    @Override
    public int compareTo(ScoredIndex that) {
        return -Double.compare(this.score, that.score);
    }

    public static List<ScoredIndex> wrap(double[] scores, double minScore) {
        List<ScoredIndex> scored = new ArrayList<>(scores.length);

        for (int idx = 0; idx < scores.length; idx++) {
            double score = scores[idx];
            if (score >= minScore) {
                scored.add(new ScoredIndex(idx, score));
            }
        }

        Collections.sort(scored);
        return scored;
    }

    public static List<ScoredIndex> wrap(double[] scores) {
        List<ScoredIndex> scored = new ArrayList<>(scores.length);

        for (int idx = 0; idx < scores.length; idx++) {
            double score = scores[idx];
            scored.add(new ScoredIndex(idx, score));
        }

        Collections.sort(scored);
        return scored;
    }

}