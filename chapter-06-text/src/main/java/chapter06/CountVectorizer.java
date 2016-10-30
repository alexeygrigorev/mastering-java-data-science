package chapter06;

import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;
import com.google.common.collect.Multiset.Entry;
import com.google.common.collect.Multisets;
import com.google.common.collect.Ordering;
import com.google.common.collect.Sets;

import smile.data.SparseDataset;
import smile.math.SparseArray;

public class CountVectorizer {

    public static class CountVectorizerBuilder {
        private int minDf = 5;
        private boolean applyIdf = true;
        private boolean sublinearTf = false;
        private boolean normalize = true;

        public CountVectorizerBuilder withMinimalDocumentFrequency(int minDf) {
            this.minDf = minDf;
            return this;
        }

        public CountVectorizerBuilder withIdfTransformation() {
            this.applyIdf = true;
            return this;
        }

        public CountVectorizerBuilder noIdfTransformation() {
            this.applyIdf = false;
            return this;
        }

        public CountVectorizerBuilder withSublinearTfTransformation() {
            this.sublinearTf = true;
            return this;
        }

        public CountVectorizerBuilder withL2Normalization() {
            this.normalize = true;
            return this;
        }

        public CountVectorizerBuilder noL2Normalization() {
            this.normalize = false;
            return this;
        }

        public CountVectorizer build() {
            return new CountVectorizer(minDf, applyIdf, sublinearTf, normalize);
        }

    }

    public static CountVectorizerBuilder create() {
        return new CountVectorizerBuilder();
    }

    public static CountVectorizer withDefaultSettings() {
        int minDf = 5;
        boolean applyIdf = true;
        boolean sublinearTf = true;
        boolean normalize = true;
        return new CountVectorizer(minDf, applyIdf, sublinearTf, normalize);
    }

    private final int minDf;
    private final boolean applyIdf;
    private final boolean sublinearTf;
    private final boolean normalize;

    private Map<String, Integer> tokenToIndex;
    private double[] idfs;

    public CountVectorizer(int minDf, boolean applyIdf, boolean sublinearTf, boolean normalize) {
        this.minDf = minDf;
        this.applyIdf = applyIdf;
        this.sublinearTf = sublinearTf;
        this.normalize = normalize;
    }

    public SparseDataset fitTransform(List<List<String>> tokens) {
        return fit(tokens).transfrom(tokens);
    }

    public CountVectorizer fit(List<List<String>> tokens) {
        Multiset<String> df = HashMultiset.create();
        tokens.forEach(list -> df.addAll(Sets.newHashSet(list)));
        Multiset<String> domainFrequency = Multisets.filter(df, p -> df.count(p) >= minDf);

        List<String> sorted = Ordering.natural().sortedCopy(domainFrequency.elementSet());
        tokenToIndex = new HashMap<>(sorted.size());
        for (int i = 0; i < sorted.size(); i++) {
            tokenToIndex.put(sorted.get(i), i);
        }

        if (applyIdf) {
            idfs = calculateIdf(domainFrequency, tokenToIndex);
        }

        return this;
    }

    private static double[] calculateIdf(Multiset<String> domainFrequency, Map<String, Integer> tokenToIndex) {
        int numDocuments = tokenToIndex.size();
        double numDocumentsLog = Math.log(numDocuments + 1);

        double[] result = new double[numDocuments];

        for (Entry<String> e : domainFrequency.entrySet()) {
            String token = e.getElement();
            double idf = numDocumentsLog - Math.log(e.getCount() + 1);
            result[tokenToIndex.get(token)] = idf;
        }

        return result;
    }

    public SparseDataset transfrom(List<List<String>> tokens) {
        int ncol = tokenToIndex.size();
        SparseDataset tfidf = new SparseDataset(ncol);

        for (int rowNo = 0; rowNo < tokens.size(); rowNo++) {
            Multiset<String> row = HashMultiset.create(tokens.get(rowNo));
            for (Entry<String> e : row.entrySet()) {
                String token = e.getElement();
                double tf = e.getCount();
                if (sublinearTf) {
                    tf = 1 + Math.log(tf);
                }

                if (!tokenToIndex.containsKey(token)) {
                    continue;
                }

                int colNo = tokenToIndex.get(token);

                if (applyIdf) {
                    double idf = idfs[colNo];
                    tfidf.set(rowNo, colNo, tf * idf);
                } else {
                    tfidf.set(rowNo, colNo, tf);
                }
            }
        }

        if (normalize) {
            tfidf.unitize();
        }

        return tfidf;
    }

    public SparseArray transfromVector(List<String> tokens) {
        SparseDataset dataset = transfrom(Arrays.asList(tokens));
        return dataset.get(0).x;
    }

    public List<String> featureNames() {
        Comparator<Map.Entry<String, Integer>> byValue = Map.Entry.comparingByValue();
        return tokenToIndex.entrySet().stream()
                .sorted(byValue)
                .map(e -> e.getKey())
                .collect(Collectors.toList());
    }

}
