package chapter09.graph;

import java.io.Serializable;

public class ScoredEdge implements Serializable {
    private long node1;
    private long node2;
    private double score;
    private double target;

    public ScoredEdge(long node1, long node2, double target) {
        this.node1 = node1;
        this.node2 = node2;
        this.target = target;
    }

    public long getNode1() {
        return node1;
    }

    public long getNode2() {
        return node2;
    }
    
    public void setScore(double score) {
        this.score = score;
    }

    public double getScore() {
        return score;
    }

    public double getTarget() {
        return target;
    }

    @Override
    public String toString() {
        return "ScoredEdge [node1=" + node1 + ", node2=" + node2 + ", score=" + score + "]";
    }
}