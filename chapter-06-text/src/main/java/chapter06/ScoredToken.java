package chapter06;

public class ScoredToken implements Comparable<ScoredToken> {
    private final String token;
    private final double score;

    public ScoredToken(String token, double score) {
        this.token = token;
        this.score = score;
    }

    @Override
    public String toString() {
        return String.format("(%s, %.4f)", token, score);
    }

    public String getToken() {
        return token;
    }

    public double getScore() {
        return score;
    }

    @Override
    public int compareTo(ScoredToken that) {
        return -Double.compare(this.score, that.score);
    }

}