package chapter06.text;

import java.io.Serializable;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

public class Document implements Serializable {

    private List<Sentence> sentences;

    public Document() {
    }

    public Document(List<Sentence> sentences) {
        this.sentences = sentences;
    }

    public List<Sentence> getSentences() {
        return sentences;
    }

    public void setSentences(List<Sentence> sentences) {
        this.sentences = sentences;
    }

    public Set<String> distinctTokens() {
        return sentences.stream()
                .flatMap(s -> s.distinctTokens().stream())
                .collect(Collectors.toSet());
    }
}
