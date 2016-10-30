package chapter06.corenlp;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import com.alexeygrigorev.rseq.BeanMatchers;
import com.alexeygrigorev.rseq.Match;
import com.alexeygrigorev.rseq.Pattern;
import com.alexeygrigorev.rseq.XMatcher;
import com.google.common.collect.ImmutableSet;

public class TokenUtils {

    private static final Set<String> ENGLISH_STOPWORDS = ImmutableSet.of("a", "an", "and", "are", "as", "at", "be",
            "but", "by", "for", "if", "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the",
            "their", "then", "there", "these", "they", "this", "to", "was", "will", "with", "what", "which");

    public static boolean isStopword(String token) {
        return ENGLISH_STOPWORDS.contains(token);
    }

    public static boolean isPunctuation(String token) {
        char first = token.charAt(0);
        return !Character.isAlphabetic(first) && !Character.isDigit(first);
    }

    private final static XMatcher<Word> ADJ = BeanMatchers.eq(Word.class, "posTag", "JJ");
    private final static XMatcher<Word> NOUN = BeanMatchers.in(Word.class, "posTag", ImmutableSet.of("NN", "NNS", "NNP"));
    private final static Pattern<Word> NOUN_PHRASE = Pattern.create(ADJ.zeroOrMore(), NOUN.oneOrMore());

    public static List<String> extractNounPhrases(List<Word> tokens) {
        List<Match<Word>> phrases = NOUN_PHRASE.find(tokens);
        return phrases.stream()
                .map(m -> m.getMatchedSubsequence())
                .map(seq -> joinLemmas(seq))
                .collect(Collectors.toList());
    }

    private static String joinLemmas(List<Word> seq) {
        return seq.stream()
                .map(Word::getLemma)
                .collect(Collectors.joining("_"));
    }
}
