package chapter09.graph;

import java.io.Serializable;

public class Edge implements Serializable {
    private final String node1;
    private final String node2;
    private final int year;

    public Edge(String node1, String node2, int year) {
        this.node1 = node1;
        this.node2 = node2;
        this.year = year;
    }

    public String getNode1() {
        return node1;
    }

    public String getNode2() {
        return node2;
    }

    public int getYear() {
        return year;
    }

    @Override
    public String toString() {
        return "Edge [node1=" + node1 + ", node2=" + node2 + ", year=" + year + "]";
    }

}