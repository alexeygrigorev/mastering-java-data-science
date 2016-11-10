package chapter06.embed;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import chapter06.ScoredToken;

public class Word2VecEmbeddings {

    public static void main(String[] args) throws IOException {
        File cached = new File("GoogleNews-vectors-negative300-java.bin");
        WordEmbeddings w2v;
        if (cached.exists()) {
            w2v = WordEmbeddings.load(cached);
        } else {
            w2v = WordEmbeddings.readWord2VecBin("GoogleNews-vectors-negative300.bin");
            w2v.save(cached);
        }

        List<String> samples = Arrays.asList("cat", "laptop", "germany", "adidas", "limit");

        for (String sample : samples) {
            System.out.println("most similar for " + sample);

            List<ScoredToken> mostSimilar = w2v.mostSimilar(sample, 10, 0.2);
            for (ScoredToken similar : mostSimilar) {
                System.out.printf(" - %.3f %s %n", similar.getScore(), similar.getToken());
            }

            System.out.println();
        }
    }
}
