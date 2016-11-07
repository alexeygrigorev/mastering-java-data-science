package chapter06.corenlp;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Properties;

import org.apache.commons.lang3.StringUtils;

import chapter06.text.TextUtils;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;

public class StanfordNlpTokenizer {
    private final ThreadLocal<StanfordCoreNLP> pipeline = createNlpPipeline();

    public List<Word> tokenize(String text) {
        if (StringUtils.isBlank(text)) {
            return Collections.emptyList();
        }

        Annotation document = new Annotation(text);
        pipeline.get().annotate(document);

        List<Word> results = new ArrayList<>();

        List<CoreLabel> tokens = document.get(TokensAnnotation.class);
        for (CoreLabel tokensInfo : tokens) {
            String token = tokensInfo.get(TextAnnotation.class);
            String lemma = tokensInfo.get(LemmaAnnotation.class);
            String pos = tokensInfo.get(PartOfSpeechAnnotation.class);

            if (TextUtils.isPunctuation(token)) {
                continue;
            }

            if (TextUtils.isStopword(token)) {
                continue;
            }

            results.add(new Word(token, lemma, pos));
        }

        return results;
    }

    private ThreadLocal<StanfordCoreNLP> createNlpPipeline() {
        return new ThreadLocal<StanfordCoreNLP>() {
            @Override
            protected synchronized StanfordCoreNLP initialValue() {
                Properties props = new Properties();
                props.put("annotators", "tokenize, ssplit, pos, lemma");
                props.put("tokenize.options", "untokenizable=noneDelete," + "strictTreebank3=true,"
                        + "ptb3Escaping=false," + "escapeForwardSlashAsterisk=false");
                props.put("ssplit.newlineIsSentenceBreak", "always");
                return new StanfordCoreNLP(props);
            }
        };
    }
}
