package chapter07.xgb;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

import org.apache.commons.lang3.Validate;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Stopwatch;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.primitives.Doubles;

import chapter07.BeanToJoinery;
import chapter07.RankedPage;
import chapter07.text.CountVectorizer;
import chapter07.text.MatrixUtils;
import chapter07.text.TruncatedSVD;
import chapter07.text.WordEmbeddings;
import joinery.DataFrame;
import no.uib.cipr.matrix.DenseMatrix;
import smile.data.SparseDataset;

public class TextFeatureExtractor implements Serializable {

    private static final Logger LOGGER = LoggerFactory.getLogger(TextFeatureExtractor.class);

    private CountVectorizer allVectorizer;
    private CountVectorizer titleVectorizer;
    private CountVectorizer headerVectorizer;
    private TruncatedSVD svdAll;
    private TruncatedSVD svdTitle;
    private WordEmbeddings glove;

    public TextFeatureExtractor fit(List<RankedPage> data) throws Exception {
        Stopwatch stopwatch;

        stopwatch = Stopwatch.createStarted();
        LOGGER.debug("tokenizing all body texts... ");
        List<List<String>> bodyText = data.parallelStream()
                .map(p -> p.getBody())
                .collect(Collectors.toList());
        LOGGER.debug("took " + stopwatch.stop());

        stopwatch = Stopwatch.createStarted();
        LOGGER.debug("tokenizing all titles... ");
        List<List<String>> titles = data.parallelStream()
                .map(p -> p.getTitle())
                .collect(Collectors.toList());
        LOGGER.debug("took " + stopwatch.stop());

        stopwatch = Stopwatch.createStarted();
        LOGGER.debug("tokenizing all headers... ");
        List<List<String>> headers = data.parallelStream()
                .map(p -> new ArrayList<>(p.getHeaders().values()))
                .collect(Collectors.toList());
        LOGGER.debug("took " + stopwatch.stop());

        List<List<String>> all = new ArrayList<>(bodyText);
        all.addAll(titles);


        stopwatch = Stopwatch.createStarted();
        LOGGER.debug("vectorizing all texts... ");
        allVectorizer = CountVectorizer.create()
                .withMinimalDocumentFrequency(5)
                .withIdfTransformation()
                .withL2Normalization()
                .withSublinearTfTransformation()
                .fit(all);
        LOGGER.debug("took " + stopwatch.stop());

        stopwatch = Stopwatch.createStarted();
        LOGGER.debug("SVD'ing all texts... ");
        svdAll = new TruncatedSVD(150, true);
        svdAll.fit(allVectorizer.transfrom(all));
        LOGGER.debug("took " + stopwatch.stop());


        stopwatch = Stopwatch.createStarted();
        LOGGER.debug("vectorizing all titles... ");
        titleVectorizer = CountVectorizer.create()
                .withMinimalDocumentFrequency(3)
                .withIdfTransformation()
                .withL2Normalization()
                .fit(titles);
        LOGGER.debug("took " + stopwatch.stop());

        stopwatch = Stopwatch.createStarted();
        LOGGER.debug("SVD'ing all titles... ");
        svdTitle = new TruncatedSVD(50, true);
        svdTitle.fit(titleVectorizer.transfrom(titles));
        LOGGER.debug("took " + stopwatch.stop());


        stopwatch = Stopwatch.createStarted();
        LOGGER.debug("vectorizing all headers... ");
        headerVectorizer = CountVectorizer.create()
                .withMinimalDocumentFrequency(3)
                .withIdfTransformation()
                .withL2Normalization()
                .fit(headers);
        LOGGER.debug("took " + stopwatch.stop());

        glove = WordEmbeddings.readGlove("glove.6B.300d.bin");

        return this;
    }

    public DataFrame<Double> transform(List<RankedPage> data) throws Exception {
        Stopwatch stopwatch;

        LOGGER.debug("tranforming the input...");

        DataFrame<Object> df = BeanToJoinery.convert(data, RankedPage.class);
        df = df.retain("url", "body", "query", "title");

//        private List<String> query;
//        private List<String> body;
//        private List<String> title;
//        private ArrayListMultimap<String, String> ;

        DataFrame<Double> result = new DataFrame<>(df.index(), Collections.emptySet());

        LOGGER.debug("tokenizing queries...");
        stopwatch = Stopwatch.createStarted();

        List<List<String>> query = castToListListString(df.col("query"));
        SparseDataset queryVectors = allVectorizer.transfrom(query);
        LOGGER.debug("took {}", stopwatch.stop());

        LOGGER.debug("tokenizing body...");
        stopwatch = Stopwatch.createStarted();
        List<List<String>> body = castToListListString(df.col("body"));
        SparseDataset bodyVectors = allVectorizer.transfrom(body);
        LOGGER.debug("took {}", stopwatch.stop());

        LOGGER.debug("computing similarity between query and raw body vectors...");
        stopwatch = Stopwatch.createStarted();
        double[] queryBodySimilarity = MatrixUtils.rowWiseSparseDot(queryVectors, bodyVectors);
        result.add("queryBodySimilarity", Doubles.asList(queryBodySimilarity));
        LOGGER.debug("took {}", stopwatch.stop());

        LOGGER.debug("computing similarity between query and body in the LSI space...");
        stopwatch = Stopwatch.createStarted();
        double[][] queryLsi = svdAll.transform(queryVectors);
        double[][] bodyLsi = svdAll.transform(bodyVectors);

        double[] queryBodyLsi = MatrixUtils.rowWiseDot(queryLsi, bodyLsi);
        result.add("queryBodyLsi", Doubles.asList(queryBodyLsi));
        LOGGER.debug("took {}", stopwatch.stop());

        LOGGER.debug("tokenizing titles...");
        stopwatch = Stopwatch.createStarted();
        List<List<String>> titles = castToListListString(df.col("title"));

        SparseDataset titleVectors = titleVectorizer.transfrom(titles);
        queryVectors = titleVectorizer.transfrom(query);
        LOGGER.debug("took {}", stopwatch.stop());

        LOGGER.debug("computing similarity between query and raw title vectors...");
        stopwatch = Stopwatch.createStarted();
        double[] queryTitleSimilarity = MatrixUtils.rowWiseSparseDot(queryVectors, titleVectors);
        result.add("queryTitleSimilarity", Doubles.asList(queryTitleSimilarity));
        LOGGER.debug("took {}", stopwatch.stop());

        LOGGER.debug("computing similarity between query and title in the LSI space...");
        stopwatch = Stopwatch.createStarted();
        double[][] titleLsi = svdTitle.transform(titleVectors);
        queryLsi = svdTitle.transform(queryVectors);
        double[] queryTitleLsi = MatrixUtils.rowWiseDot(queryLsi, titleLsi);
        result.add("queryTitleLsi", Doubles.asList(queryTitleLsi));
        LOGGER.debug("took {}", stopwatch.stop());

        LOGGER.debug("computing glove features for titles...");
        stopwatch = Stopwatch.createStarted();
        gloveSimilarityDistribution(glove, query, titles, "queryTitlesGlove", result);
        LOGGER.debug("took {}", stopwatch.stop());

        LOGGER.debug("tokenizing headers...");
        stopwatch = Stopwatch.createStarted();
        @SuppressWarnings("unchecked")
        List<List<String>> headers = df.col("headers").stream()
                .map(h -> (ArrayListMultimap<String, String>) h)
                .map(h -> new ArrayList<>(h.values()))
                .collect(Collectors.toList());

        SparseDataset headerVectors = headerVectorizer.transfrom(headers);
        queryVectors = headerVectorizer.transfrom(query);
        LOGGER.debug("took {}", stopwatch.stop());

        LOGGER.debug("computing similarity between query and raw headers vectors...");
        stopwatch = Stopwatch.createStarted();
        double[] queryHeaderSimilarity = MatrixUtils.rowWiseSparseDot(queryVectors, headerVectors);
        result.add("queryHeaderSimilarity", Doubles.asList(queryHeaderSimilarity));
        LOGGER.debug("took {}", stopwatch.stop());

        LOGGER.debug("computing individual headers featurse...");
        stopwatch = Stopwatch.createStarted();

        String[] headerTags = { "h1", "h2", "h3" };
        for (String headerTag : headerTags) {
            @SuppressWarnings("unchecked")
            List<List<String>> header = df.col("headers").stream()
                    .map(h -> (ArrayListMultimap<String, String>) h)
                    .map(h -> h.get(headerTag))
                    .collect(Collectors.toList());

            headerVectors = headerVectorizer.transfrom(header);

            queryHeaderSimilarity = MatrixUtils.rowWiseSparseDot(queryVectors, headerVectors);
            result.add("queryHeaderSimilarity_" + headerTag, Doubles.asList(queryHeaderSimilarity));

            gloveSimilarityDistribution(glove, query, header, "queryHeader" + headerTag + "Glove", result);
        }
        LOGGER.debug("took {}", stopwatch.stop());

        return result;
    }

    private static List<List<String>> castToListListString(List<Object> input) {
        List<?> untyped = input;
        @SuppressWarnings("unchecked")
        List<List<String>> retyped = (List<List<String>>) untyped;
        return retyped;
    }

    private static void gloveSimilarityDistribution(WordEmbeddings glove, List<List<String>> query,
            List<List<String>> text, String featureNamePrefix, DataFrame<Double> dataFrame) {
        Validate.isTrue(query.size() == text.size(), "sizes of lists do not match: %d != %d", query.size(),
                text.size());

        int size = query.size();

        List<Double> mins = new ArrayList<>(size);
        List<Double> means = new ArrayList<>(size);
        List<Double> maxs = new ArrayList<>(size);
        List<Double> stds = new ArrayList<>(size);
        List<Double> avgCos = new ArrayList<>(size);

        for (int i = 0; i < size; i++) {
            double[][] queryEmbed = wordsToVec(glove, query.get(i));
            double[][] textEmbed = wordsToVec(glove, text.get(i));

            if (queryEmbed.length == 0 || textEmbed.length == 0) {
                mins.add(Double.NaN);
                means.add(Double.NaN);
                maxs.add(Double.NaN);
                stds.add(Double.NaN);
                avgCos.add(Double.NaN);
                continue;
            }

            double[] similarities = similarities(queryEmbed, textEmbed);
            DescriptiveStatistics stats = new DescriptiveStatistics(similarities);

            mins.add(stats.getMin());
            means.add(stats.getMean());
            maxs.add(stats.getMax());
            stds.add(stats.getStandardDeviation());

            double[] avgQuery = averageVector(queryEmbed);
            double[] avgText = averageVector(textEmbed);
            double cos = dot(avgQuery, avgText);
            avgCos.add(cos);
        }

        dataFrame.add(featureNamePrefix + "_min", mins);
        dataFrame.add(featureNamePrefix + "_mean", means);
        dataFrame.add(featureNamePrefix + "_max", maxs);
        dataFrame.add(featureNamePrefix + "_std", stds);
        dataFrame.add(featureNamePrefix + "_avg_cos", avgCos);
    }

    private static double dot(double[] v1, double[] v2) {
        ArrayRealVector vec1 = new ArrayRealVector(v1, false);
        ArrayRealVector vec2 = new ArrayRealVector(v2, false);
        return vec1.dotProduct(vec2);
    }

    private static double[] averageVector(double[][] rows) {
        ArrayRealVector acc = new ArrayRealVector(rows[0], true);

        for (int i = 1; i < rows.length; i++) {
            ArrayRealVector vec = new ArrayRealVector(rows[0], false);
            acc.combineToSelf(1.0, 1.0, vec);
        }

        double norm = acc.getNorm();
        acc.mapDivideToSelf(norm);
        return acc.getDataRef();
    }

    private static double[] similarities(double[][] m1, double[][] m2) {
        DenseMatrix M1 = new DenseMatrix(m1);
        DenseMatrix M2 = new DenseMatrix(m2);

        DenseMatrix M1M2 = new DenseMatrix(M1.numRows(), M2.numRows());
        M1.transBmult(M2, M1M2);

        return M1M2.getData();
    }

    private static double[][] wordsToVec(WordEmbeddings glove, List<String> tokens) {
        List<double[]> vectors = new ArrayList<>(tokens.size());
        for (String token : tokens) {
            Optional<double[]> vector = glove.representation(token);
            if (vector.isPresent()) {
                vectors.add(vector.get());
            }
        }

        int nrows = vectors.size();
        double[][] result = new double[nrows][];
        for (int i = 0; i < nrows; i++) {
            result[i] = vectors.get(i);
        }

        return result;
    }

}
