package chapter06;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;
import com.google.common.collect.Multisets;
import com.google.common.collect.Ordering;
import com.google.common.collect.Sets;
import com.google.common.collect.Multiset.Entry;

import smile.data.SparseDataset;

public class CoOccurrenceMatrix {

    private final int minDf;
    private Map<String, Integer> tokenToIndex;

    public CoOccurrenceMatrix(int minDf) {
        this.minDf = minDf;
    }

    public CoOccurrenceMatrix fit(List<List<String>> tokens) {
        Multiset<String> df = HashMultiset.create();
        tokens.forEach(list -> df.addAll(Sets.newHashSet(list)));
        Multiset<String> domainFrequency = Multisets.filter(df, p -> df.count(p) >= minDf);

        List<String> sorted = Ordering.natural().sortedCopy(domainFrequency.elementSet());
        tokenToIndex = new HashMap<>(sorted.size());
        for (int i = 0; i < sorted.size(); i++) {
            tokenToIndex.put(sorted.get(i), i);
        }

        return this;
    }

    public SparseDataset transform(List<List<String>> tokens) {
        int ncol = tokenToIndex.size();
        SparseDataset result = new SparseDataset(ncol);

        for (int rowNo = 0; rowNo < tokens.size(); rowNo++) {
            List<String> list = tokens.get(rowNo);
            for (String token : list) {
                if (!tokenToIndex.containsKey(token)) {
                    continue;
                }

                int colNo = tokenToIndex.get(token);
                result.set(i, y, weight);
                
            }
        }
    }
}
