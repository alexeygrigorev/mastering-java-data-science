package chapter06.project;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

import org.apache.commons.lang3.Validate;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import com.google.common.base.Stopwatch;
import com.google.common.primitives.Doubles;

import chapter06.MatrixUtils;
import chapter06.TruncatedSVD;
import chapter06.embed.WordEmbeddings;
import chapter06.project.PrepareData.HtmlDocument;
import chapter06.project.PrepareData.LabeledQueryDocumentPair;
import chapter06.text.CountVectorizer;
import chapter06.text.TextUtils;
import joinery.DataFrame;
import no.uib.cipr.matrix.DenseMatrix;
import smile.data.SparseDataset;

public class FeatureExtractor {

    private CountVectorizer allVectorizer;
    private CountVectorizer titleVectorizer;
    private CountVectorizer headerVectorizer;
    private TruncatedSVD svdAll;
    private TruncatedSVD svdTitle;

    public FeatureExtractor fit(List<LabeledQueryDocumentPair> data) {
        Stopwatch stopwatch;

        stopwatch = Stopwatch.createStarted();
        System.out.print("tokenizing all body texts... ");
        List<List<String>> bodyText = data.parallelStream()
                .map(p -> p.getDocument().getBodyText())
                .map(t -> TextUtils.tokenizeFilter(t))
                .collect(Collectors.toList());
        System.out.println("took " + stopwatch.stop());

        stopwatch = Stopwatch.createStarted();
        System.out.print("tokenizing all titles... ");
        List<List<String>> titles = data.parallelStream()
                .map(p -> p.getDocument().getTitle())
                .map(t -> TextUtils.tokenizeFilter(t))
                .collect(Collectors.toList());
        System.out.println("took " + stopwatch.stop());

        stopwatch = Stopwatch.createStarted();
        System.out.print("tokenizing all headers... ");
        List<List<String>> headers = data.parallelStream()
                .flatMap(p -> p.getDocument().getHeaders().values().stream())
                .map(t -> TextUtils.tokenizeFilter(t))
                .collect(Collectors.toList());
        System.out.println("took " + stopwatch.stop());

        List<List<String>> all = new ArrayList<>(bodyText);
        all.addAll(titles);


        stopwatch = Stopwatch.createStarted();
        System.out.print("vectorizing all texts... ");
        allVectorizer = CountVectorizer.create()
                .withMinimalDocumentFrequency(5)
                .withIdfTransformation()
                .withL2Normalization()
                .withSublinearTfTransformation()
                .fit(all);
        System.out.println("took " + stopwatch.stop());

        stopwatch = Stopwatch.createStarted();
        System.out.print("SVD'ing all texts... ");
        svdAll = new TruncatedSVD(150, true);
        svdAll.fit(allVectorizer.transfrom(all));
        System.out.println("took " + stopwatch.stop());


        stopwatch = Stopwatch.createStarted();
        System.out.print("vectorizing all titles... ");
        titleVectorizer = CountVectorizer.create()
                .withMinimalDocumentFrequency(3)
                .withIdfTransformation()
                .withL2Normalization()
                .fit(titles);
        System.out.println("took " + stopwatch.stop());

        stopwatch = Stopwatch.createStarted();
        System.out.print("SVD'ing all titles... ");
        svdTitle = new TruncatedSVD(50, true);
        svdTitle.fit(titleVectorizer.transfrom(titles));
        System.out.println("took " + stopwatch.stop());


        stopwatch = Stopwatch.createStarted();
        System.out.print("vectorizing all headers... ");
        headerVectorizer = CountVectorizer.create()
                .withMinimalDocumentFrequency(3)
                .withIdfTransformation()
                .withL2Normalization()
                .fit(headers);
        System.out.println("took " + stopwatch.stop());

        return this;
    }

    public DataFrame<Number> transform(List<LabeledQueryDocumentPair> data) throws Exception {
        DataFrame<Object> dataFrame = BeanToJoinery.convert(data, LabeledQueryDocumentPair.class);
        List<HtmlDocument> documents = dataFrame.cast(HtmlDocument.class).col("document");

        // cannot use parallel streams here because the order is important
        List<List<String>> query = dataFrame.col("query").stream()
                .map(q -> (String) q)
                .map(q -> TextUtils.tokenizeFilter(q))
                .collect(Collectors.toList());
        SparseDataset queryVectors = allVectorizer.transfrom(query);

        List<List<String>> body = documents.stream()
                .map(d -> d.getBodyText())
                .map(t -> TextUtils.tokenizeFilter(t))
                .collect(Collectors.toList());
        SparseDataset bodyVectors = allVectorizer.transfrom(body);

        double[] queryBodySimilarity = MatrixUtils.rowWiseSparseDot(queryVectors, bodyVectors);
        dataFrame.add("queryBodySimilarity", arrayToList(queryBodySimilarity));

        double[][] queryLsi = svdAll.transform(queryVectors);
        double[][] bodyLsi = svdAll.transform(bodyVectors);

        double[] queryBodyLsi = MatrixUtils.rowWiseDot(queryLsi, bodyLsi);
        dataFrame.add("queryBodyLsi", arrayToList(queryBodyLsi));


        List<List<String>> titles = documents.stream()
                .map(d -> d.getTitle())
                .map(t -> TextUtils.tokenizeFilter(t))
                .collect(Collectors.toList());

        SparseDataset titleVectors = titleVectorizer.transfrom(titles);
        queryVectors = titleVectorizer.transfrom(query);

        double[] queryTitleSimilarity = MatrixUtils.rowWiseSparseDot(queryVectors, titleVectors);
        dataFrame.add("queryTitleSimilarity", arrayToList(queryTitleSimilarity));

        double[][] titleLsi = svdTitle.transform(titleVectors);
        queryLsi = svdTitle.transform(queryVectors);

        WordEmbeddings glove = WordEmbeddings.load("glove.6B.300d.bin");
        gloveSimilarityDistribution(glove, query, titles, "queryTitlesGlove", dataFrame);

        double[] queryTitleLsi = MatrixUtils.rowWiseDot(queryLsi, titleLsi);
        dataFrame.add("queryTitleLsi", arrayToList(queryTitleLsi));

        List<List<String>> headers = documents.stream()
                .map(d -> d.getHeaders())
                .map(h -> String.join(" ", h.values()))
                .map(t -> TextUtils.tokenizeFilter(t))
                .collect(Collectors.toList());

        SparseDataset headerVectors = headerVectorizer.transfrom(headers);
        queryVectors = headerVectorizer.transfrom(query);

        double[] queryHeaderSimilarity = MatrixUtils.rowWiseSparseDot(queryVectors, headerVectors);
        dataFrame.add("queryHeaderSimilarity", arrayToList(queryHeaderSimilarity));

        String[] headerTags = { "h1", "h2", "h3" };
        for (String headerTag : headerTags) {
            List<List<String>> header = documents.stream()
                    .map(d -> d.getHeaders())
                    .map(h -> String.join(" ", h.get(headerTag)))
                    .map(t -> TextUtils.tokenizeFilter(t))
                    .collect(Collectors.toList());

            headerVectors = headerVectorizer.transfrom(header);

            queryHeaderSimilarity = MatrixUtils.rowWiseSparseDot(queryVectors, headerVectors);
            dataFrame.add("queryHeaderSimilarity_" + headerTag, arrayToList(queryHeaderSimilarity));

            gloveSimilarityDistribution(glove, query, header, "queryHeader" + headerTag + "Glove", dataFrame);
        }

        DataFrame<Object> result = dataFrame.drop("document", "query", "train");
        return result.cast(Number.class);
    }

    private static void gloveSimilarityDistribution(WordEmbeddings glove, List<List<String>> query,
            List<List<String>> text, String featureNamePrefix, DataFrame<Object> dataFrame) {
        Validate.isTrue(query.size() == text.size(), "sizes of lists do not match: %d != %d", query.size(),
                text.size());

        int size = query.size();

        List<Object> mins = new ArrayList<>(size);
        List<Object> means = new ArrayList<>(size);
        List<Object> maxs = new ArrayList<>(size);
        List<Object> stds = new ArrayList<>(size);
        List<Object> avgCos = new ArrayList<>(size);

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

    private static List<Object> arrayToList(double[] queryBodyLsi) {
        @SuppressWarnings("unchecked")
        List<Object> queryList = (List<Object>) (List<?>) Doubles.asList(queryBodyLsi);
        return queryList;
    }


}
