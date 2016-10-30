package chapter06.ownir;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;

import chapter06.CountVectorizer;
import chapter06.MatrixUtils;
import chapter06.TextUtils;
import smile.data.SparseDataset;
import smile.math.SparseArray;

public class IRSimpleText {

    public static void main(String[] args) throws Exception {
        Path path = Paths.get("data/simple-text.txt");

        List<List<String>> documents = Files.lines(path, StandardCharsets.UTF_8)
                .map(line -> TextUtils.tokenize(line))
                .map(line -> TextUtils.removeStopwords(line))
                .collect(Collectors.toList());

        documents.forEach(System.out::println);


        CountVectorizer vectorizer = CountVectorizer.create()
                .withMinimalDocumentFrequency(2)
                .withSublinearTfTransformation()
                .build();

        SparseDataset docVectors = vectorizer.fitTransform(documents);

        List<String> featureNames = vectorizer.featureNames();
        docVectors.forEach(e -> {
            e.x.forEach(i -> System.out.printf("(%s, %.3f) ", featureNames.get(i.i), i.x));
            System.out.println();
        });

        List<String> query = TextUtils.tokenize("the probabilistic interpretation of tf-idf");

        SparseArray queryVector = vectorizer.transfromVector(query);

        double[] scores = MatrixUtils.vectorSimilarity(docVectors, queryVector);

        double minScore = 0.2;
        List<ScoredIndex> scored = ScoredIndex.wrap(scores, minScore);

        for (ScoredIndex doc : scored) {
            System.out.printf("> %.4f ", doc.getScore());

            List<String> document = documents.get(doc.getIndex());
            System.out.println(String.join(" ", document));
        }
    }

}
