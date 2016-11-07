package chapter06.embed;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Properties;

import org.apache.commons.lang3.StringUtils;

import chapter06.text.Document;
import chapter06.text.Sentence;
import chapter06.text.TextUtils;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

public class StanfordNlpSentenceTokenizer {
    private final ThreadLocal<StanfordCoreNLP> pipeline = createNlpPipeline();

    public Document tokenize(String text) {
        if (StringUtils.isBlank(text)) {
            return new Document(Collections.emptyList());
        }

        Annotation document = new Annotation(text);
        pipeline.get().annotate(document);

        List<Sentence> sentencesResult = new ArrayList<>();

        List<CoreMap> sentences = document.get(SentencesAnnotation.class);
        for (CoreMap sentence : sentences) {
            List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
            List<String> tokensResult = new ArrayList<>();

            for (CoreLabel tokensInfo : tokens) {
                String token = tokensInfo.get(TextAnnotation.class);
                String lemma = tokensInfo.get(LemmaAnnotation.class);

                if (TextUtils.isPunctuation(token)) {
                    continue;
                }

                if (TextUtils.isStopword(token)) {
                    continue;
                }

                if (TextUtils.isStopword(lemma)) {
                    continue;
                }

                if (lemma.length() <= 2) {
                    continue;
                }

                lemma = lemma.toLowerCase();
                tokensResult.add(lemma);
            }

            if (!tokensResult.isEmpty()) {
                sentencesResult.add(new Sentence(tokensResult));
            }
        }

        return new Document(sentencesResult);
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
