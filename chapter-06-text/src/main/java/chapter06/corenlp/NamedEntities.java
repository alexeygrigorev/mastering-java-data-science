package chapter06.corenlp;

import static com.alexeygrigorev.rseq.Matchers.x;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Properties;
import java.util.stream.Collectors;

import org.apache.commons.lang3.StringUtils;

import com.alexeygrigorev.rseq.Pattern;
import com.alexeygrigorev.rseq.XMatcher;

import chapter06.text.TextUtils;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;

public class NamedEntities {
    private final StanfordCoreNLP pipeline = createNlpPipeline();

    public List<Word> tokenize(String text) {
        if (StringUtils.isBlank(text)) {
            return Collections.emptyList();
        }

        Annotation document = new Annotation(text);
        pipeline.annotate(document);

        List<Word> results = new ArrayList<>();

        List<CoreLabel> tokens = document.get(TokensAnnotation.class);
        for (CoreLabel tokensInfo : tokens) {
            String token = tokensInfo.get(TextAnnotation.class);
            String lemma = tokensInfo.get(LemmaAnnotation.class);
            String pos = tokensInfo.get(PartOfSpeechAnnotation.class);
            String ner = tokensInfo.get(NamedEntityTagAnnotation.class);

            if (TextUtils.isPunctuation(token)) {
                continue;
            }

            if (TextUtils.isStopword(token)) {
                continue;
            }

            results.add(new Word(token, lemma, pos, ner));
        }

        return results;
    }

    private StanfordCoreNLP createNlpPipeline() {
        Properties props = new Properties();
        props.put("annotators", "tokenize, ssplit, pos, lemma, ner");
        props.put("tokenize.options", "untokenizable=noneDelete," + "strictTreebank3=true," + "ptb3Escaping=false,"
                + "escapeForwardSlashAsterisk=false");
        props.put("ssplit.newlineIsSentenceBreak", "always");
        return new StanfordCoreNLP(props);
    }

    public static void main(String[] args) {
        NamedEntities tokenizer = new NamedEntities();
        List<Word> tokens = tokenizer.tokenize("My name is Justin Bieber, I live in Canada.");
        System.out.println(tokens);

        XMatcher<Word> notO = x(w -> !w.getNer().equals("O")); 
        Pattern<Word> namedEntities = Pattern.create(notO.oneOrMore());

        tokens = namedEntities.replaceToOne(tokens, match -> {
            List<Word> matched = match.getMatchedSubsequence();
            String lemmas = matched.stream().map(Word::getLemma)
                    .map(String::toLowerCase)
                    .collect(Collectors.joining("_"));
            return new Word(lemmas, lemmas, "NE", matched.get(0).getNer());
        });

        System.out.println(tokens);
    }
}
