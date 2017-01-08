package chapter09.graph;

import java.io.IOException;
import java.util.Arrays;

import it.unimi.dsi.logging.ProgressLogger;
import it.unimi.dsi.webgraph.BVGraph;
import it.unimi.dsi.webgraph.NodeIterator;

public class CommonCrawlGraph {

    public static void main(String[] args) throws IOException {
        String baseName = "/home/agrigorev/tmp/data/cc/hostgraph";
        BVGraph graph = BVGraph.loadMapped(baseName, new ProgressLogger());

        NodeIterator nodes = graph.nodeIterator();

        while (nodes.hasNext()) {
            int next = nodes.nextInt();
            int[] others1 = nodes.successorArray();
            System.out.println(Arrays.toString(others1));
            break;
        }
    }
}
