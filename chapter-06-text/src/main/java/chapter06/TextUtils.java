package chapter06;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import com.google.common.collect.ImmutableSet;

public class TextUtils {

    private static final Set<String> EN_STOPWORDS = ImmutableSet.of("a", "the", "is", "are", "am", "be", "was", "and",
            "as", "by", "for", "in", "to", "of", "but");

    public static List<String> tokenize(String line) {
        Pattern pattern = Pattern.compile("\\W+");
        String[] split = pattern.split(line.toLowerCase());
        return Arrays.asList(split);
    }

    public static List<String> removeStopwords(List<String> line) {
        return removeStopwords(line, EN_STOPWORDS);
    }

    public static List<String> removeStopwords(List<String> line, Set<String> stopwords) {
        return line.stream().filter(token -> !stopwords.contains(token)).collect(Collectors.toList());
    }

    public static List<String> ngrams(List<String> tokens, int minN, int maxN) {
        List<String> result = new ArrayList<>();

        for (int n = minN; n <= maxN; n++) {
            List<String> ngrams = ngrams(tokens, n);
            result.addAll(ngrams);
        }

        return result;
    }

    public static List<String> ngrams(List<String> tokens, int n) {
        int size = tokens.size();
        if (n > size) {
            return Collections.emptyList();
        }

        List<String> result = new ArrayList<>();

        for (int i = 0; i < size - n + 1; i++) {
            List<String> sublist = tokens.subList(i, i + n);
            result.add(String.join("_", sublist));
        }

        return result;
    }

}