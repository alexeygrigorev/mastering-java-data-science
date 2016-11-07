package chapter06.embed;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import chapter06.MatrixUtils;
import chapter06.Projections;
import chapter06.ScoredIndex;
import chapter06.ScoredToken;
import smile.data.SparseDataset;
import smile.math.matrix.SingularValueDecomposition;

public class WordEmbeddings implements Serializable {

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
}
