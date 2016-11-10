package chapter06.embed;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import chapter06.ScoredToken;

public class GloveEmbeddings {

    public static void main(String[] args) throws IOException {
        WordEmbeddings glove = WordEmbeddings.readGlove("glove.6B.300d.txt");
        List<String> samples = Arrays.asList("cat", "laptop", "germany", "adidas", "limit");

        for (String sample : samples) {
            System.out.println("most similar for " + sample);

            List<ScoredToken> mostSimilar = glove.mostSimilar(sample, 10, 0.2);
            for (ScoredToken similar : mostSimilar) {
                System.out.printf(" - %.3f %s %n", similar.getScore(), similar.getToken());
            }

            System.out.println();
        }

        glove.save("glove.6B.300d.bin");
        WordEmbeddings.load("glove.6B.300d.bin");
    }
}
