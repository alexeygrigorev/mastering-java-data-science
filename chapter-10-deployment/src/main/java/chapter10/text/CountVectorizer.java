package chapter10.text;

import java.io.Serializable;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;
import com.google.common.collect.Multiset.Entry;
import com.google.common.collect.Multisets;
import com.google.common.collect.Ordering;
import com.google.common.collect.Sets;

import smile.data.SparseDataset;
import smile.math.SparseArray;

public class CountVectorizer implements Serializable {

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

        public CountVectorizer fit(List<List<String>> documents) {
            CountVectorizer vectorizer = build();
            vectorizer.fit(documents);
            return vectorizer;
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
    private List<String> vocabulary;
    private double[] idfs;

    public CountVectorizer(int minDf, boolean applyIdf, boolean sublinearTf, boolean normalize) {
        this.minDf = minDf;
        this.applyIdf = applyIdf;
        this.sublinearTf = sublinearTf;
        this.normalize = normalize;
    }

    public SparseDataset fitTransform(List<List<String>> documents) {
        return fit(documents).transfrom(documents);
    }

    public CountVectorizer fit(List<List<String>> documents) {
        Multiset<String> df = HashMultiset.create();
        documents.forEach(list -> df.addAll(Sets.newHashSet(list)));
        Multiset<String> docFrequency = Multisets.filter(df, p -> df.count(p) >= minDf);

        vocabulary = Ordering.natural().sortedCopy(docFrequency.elementSet());
        tokenToIndex = new HashMap<>(vocabulary.size());
        for (int i = 0; i < vocabulary.size(); i++) {
            tokenToIndex.put(vocabulary.get(i), i);
        }

        if (applyIdf) {
            idfs = calculateIdf(docFrequency, tokenToIndex, documents.size());
        }

        return this;
    }

    private static double[] calculateIdf(Multiset<String> domainFrequency, Map<String, Integer> tokenToIndex,
            int numDocuments) {
        double numDocumentsLog = Math.log(numDocuments + 1);

        double[] result = new double[tokenToIndex.size()];

        for (Entry<String> e : domainFrequency.entrySet()) {
            String token = e.getElement();
            double idf = numDocumentsLog - Math.log(e.getCount() + 1);
            result[tokenToIndex.get(token)] = idf;
        }

        return result;
    }

    public SparseDataset transfrom(List<List<String>> documents) {
        int nrow = documents.size();
        int ncol = tokenToIndex.size();

        SparseDataset tfidf = new SparseDataset(ncol);

        for (int rowNo = 0; rowNo < nrow; rowNo++) {
            tfidf.set(rowNo, 0, 0.0);

            Multiset<String> row = HashMultiset.create(documents.get(rowNo));

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
        SparseDataset dataset = transfrom(Collections.singletonList(tokens));
        return dataset.get(0).x;
    }

    public List<String> vocabulary() {
        return vocabulary;
    }

}
