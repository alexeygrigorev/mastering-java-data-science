package chapter06.text;

import java.io.Serializable;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class Sentence implements Serializable {

    private List<String> tokens;

    public Sentence(List<String> tokens) {
        this.tokens = tokens;
    }

    public List<String> getTokens() {
        return tokens;
    }

    public void setTokens(List<String> tokens) {
        this.tokens = tokens;
    }

    public Set<String> distinctTokens() {
        return new HashSet<>(tokens);
    }

    @Override
    public String toString() {
        return String.join(" ", tokens);
    }
}
