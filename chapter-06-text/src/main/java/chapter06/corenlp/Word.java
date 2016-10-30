package chapter06.corenlp;

public class Word {

    private final String token;
    private final String lemma;
    private final String posTag;
    private final String ner;

    public Word(String token, String lemma, String posTag) {
        this(token, lemma, posTag, "X");
    }

    public Word(String token, String lemma, String posTag, String ner) {
        this.token = token;
        this.lemma = lemma;
        this.posTag = posTag;
        this.ner = ner;
    }

    public String getToken() {
        return token;
    }

    public String getLemma() {
        return lemma;
    }

    public String getPosTag() {
        return posTag;
    }

    public String getNer() {
        return ner;
    }

    @Override
    public String toString() {
        return token + "_" + lemma + "_" + posTag + "_" + ner;
    }

}
