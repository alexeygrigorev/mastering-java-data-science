package chapter06.embed;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.commons.lang3.SerializationUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Stopwatch;

import chapter06.MatrixUtils;
import chapter06.Projections;
import chapter06.ScoredIndex;
import chapter06.ScoredToken;
import smile.data.SparseDataset;
import smile.math.matrix.SingularValueDecomposition;

public class WordEmbeddings implements Serializable {

    private static final Logger LOGGER = LoggerFactory.getLogger(WordEmbeddings.class);

    public static enum DimRedMethod {
        SVD, RANDOM_PROJECTION;
    }

    private final double[][] embeddings;
    private final List<String> indexToToken;
    private final Map<String, Integer> tokenToIndex;

    public WordEmbeddings(double[][] embeddings, List<String> indexToToken, Map<String, Integer> tokenToIndex) {
        this.embeddings = embeddings;
        this.indexToToken = indexToToken;
        this.tokenToIndex = tokenToIndex;
    }

    public WordEmbeddings(double[][] embeddings, List<String> indexToToken) {
        this(embeddings, indexToToken, tokenToIndex(indexToToken));
    }

    public List<ScoredToken> mostSimilar(String sample, int topK, double minSimilarity) {
        int idx = tokenToIndex.get(sample);
        double[] vector = embeddings[idx];

        double[] sims = MatrixUtils.vectorSimilarity(embeddings, vector);

        List<ScoredIndex> scored = ScoredIndex.wrap(sims, minSimilarity);
        List<ScoredIndex> top = safeTop(scored, topK + 1);

        List<ScoredToken> result = new ArrayList<>(top.size() - 1);

        for (ScoredIndex si : top.subList(1, top.size())) {
            String token = indexToToken.get(si.getIndex());
            double score = si.getScore();
            result.add(new ScoredToken(token, score));
        }

        return result;
    }

    public static WordEmbeddings create(PmiCoOccurrenceMatrix pmiCooc, int dimensionality, DimRedMethod dimRed) {
        double[][] embedding;

        SparseDataset matrix = pmiCooc.getPmiMatrix();
        if (dimRed == DimRedMethod.SVD) {
            embedding = svd(matrix, dimensionality);
        } else if (dimRed == DimRedMethod.RANDOM_PROJECTION) {
            embedding = randomProjection(matrix, dimensionality);
        } else {
            throw new IllegalArgumentException("unexpected value for dimRed=" + dimRed);
        }

        Map<String, Integer> tokenToIndex = pmiCooc.getTokenToIndex();
        List<String> indexToToken = pmiCooc.getIndexToToken();

        return new WordEmbeddings(embedding, indexToToken, tokenToIndex);
    }

    private static double[][] randomProjection(SparseDataset matrix, int outputDimension) {
        double[][] basis = Projections.randomProjection(matrix.ncols(), outputDimension, 1);
        double[][] projection = Projections.project(matrix, basis);
        return MatrixUtils.l2RowNormalize(projection);
    }

    private static double[][] svd(SparseDataset matrix, int outputDimension) {
        SingularValueDecomposition svd = SingularValueDecomposition.decompose(matrix.toSparseMatrix(), outputDimension);
        double[][] basis = svd.getV();
        double[][] projection = Projections.project(matrix, basis);
        return MatrixUtils.l2RowNormalize(projection);
    }

    private static List<ScoredIndex> safeTop(List<ScoredIndex> scored, int topK) {
        if (scored.size() <= topK) {
            return scored;
        }
        return scored.subList(0, topK);
    }

    private static Map<String, Integer> tokenToIndex(List<String> indexToToken) {
        Map<String, Integer> tokenToIndex = new HashMap<>(indexToToken.size());
        for (int i = 0; i < indexToToken.size(); i++) {
            tokenToIndex.put(indexToToken.get(i), i);
        }
        return tokenToIndex;
    }

    public static WordEmbeddings load(File file) throws IOException {
        Stopwatch stopwatch = Stopwatch.createStarted();
        try (InputStream is = Files.newInputStream(file.toPath())) {
            try (BufferedInputStream bis = new BufferedInputStream(is)) {
                return SerializationUtils.deserialize(bis);
            }
        } finally {
            LOGGER.debug("loading embeddings from {} took {}", file, stopwatch.stop());
        }
    }

    public static WordEmbeddings load(String file) throws IOException {
        return load(new File(file));
    }

    public void save(String file) throws IOException {
        save(new File(file));
    }

    public void save(File file) throws IOException {
        Stopwatch stopwatch = Stopwatch.createStarted();
        try (OutputStream os = Files.newOutputStream(file.toPath())) {
            try (BufferedOutputStream bos = new BufferedOutputStream(os)) {
                SerializationUtils.serialize(this, bos);
            }
        } finally {
            LOGGER.debug("saving embeddings to {} took {}", file, stopwatch.stop());
        }
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

        List<String> indexToToken = new ArrayList<>(pairs.size());
        double[][] embeddings = new double[pairs.size()][];

        for (int i = 0; i < pairs.size(); i++) {
            Pair<String, double[]> pair = pairs.get(i);
            indexToToken.add(pair.getLeft());
            embeddings[i] = pair.getRight();
        }

        embeddings = MatrixUtils.l2RowNormalize(embeddings);
        WordEmbeddings result = new WordEmbeddings(embeddings, indexToToken);
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
