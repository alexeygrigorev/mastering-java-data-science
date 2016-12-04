package chapter07.text;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Stopwatch;

public class WordEmbeddings implements Serializable {

    private static final Logger LOGGER = LoggerFactory.getLogger(WordEmbeddings.class);

    public static enum DimRedMethod {
        SVD, RANDOM_PROJECTION;
    }

    private final double[][] embeddings;
    private final List<String> vocabulary;
    private final Map<String, Integer> tokenToIndex;

    public WordEmbeddings(double[][] embeddings, List<String> vocabulary, Map<String, Integer> tokenToIndex) {
        this.embeddings = embeddings;
        this.vocabulary = vocabulary;
        this.tokenToIndex = tokenToIndex;
    }

    public WordEmbeddings(double[][] embeddings, List<String> vocabulary) {
        this(embeddings, vocabulary, tokenToIndex(vocabulary));
    }

    public Optional<double[]> representation(String token) {
        if (tokenToIndex.containsKey(token)) {
            int idx = tokenToIndex.get(token);
            double[] vector = embeddings[idx];
            return Optional.of(vector);
        }

        return Optional.empty();
    }

    public List<String> getVocabulary() {
        return vocabulary;
    }

    private static Map<String, Integer> tokenToIndex(List<String> indexToToken) {
        Map<String, Integer> tokenToIndex = new HashMap<>(indexToToken.size());
        for (int i = 0; i < indexToToken.size(); i++) {
            tokenToIndex.put(indexToToken.get(i), i);
        }
        return tokenToIndex;
    }

    public static WordEmbeddings readGlove(String file) throws IOException {
        return readGlove(new File(file));
    }

    public static WordEmbeddings readGlove(File file) throws IOException {
        Stopwatch stopwatch = Stopwatch.createStarted();

        List<Pair<String, double[]>> pairs = Files.lines(file.toPath(), StandardCharsets.UTF_8)
                .parallel()
                .map(String::trim)
                .filter(StringUtils::isNotEmpty)
                .map(line -> parseGloveTextLine(line))
                .collect(Collectors.toList());

        List<String> vocabulary = new ArrayList<>(pairs.size());
        double[][] embeddings = new double[pairs.size()][];

        for (int i = 0; i < pairs.size(); i++) {
            Pair<String, double[]> pair = pairs.get(i);
            vocabulary.add(pair.getLeft());
            embeddings[i] = pair.getRight();
        }

        embeddings = MatrixUtils.l2RowNormalize(embeddings);
        WordEmbeddings result = new WordEmbeddings(embeddings, vocabulary);
        LOGGER.debug("loading GloVe embeddings from {} took {}", file, stopwatch.stop());
        return result;
    }

    private static Pair<String, double[]> parseGloveTextLine(String line) {
        List<String> split = Arrays.asList(line.split(" "));
        String token = split.get(0);
        double[] vector = split.subList(1, split.size()).stream()
                .mapToDouble(Double::parseDouble).toArray();
        return ImmutablePair.of(token, vector);
    }
}
