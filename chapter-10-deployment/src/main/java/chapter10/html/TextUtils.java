package chapter10.html;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.StringEscapeUtils;
import org.archive.io.ArchiveRecord;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;

import com.google.common.collect.ImmutableSet;
import com.optimaize.langdetect.LanguageDetector;
import com.optimaize.langdetect.LanguageDetectorBuilder;
import com.optimaize.langdetect.i18n.LdLocale;
import com.optimaize.langdetect.ngram.NgramExtractors;
import com.optimaize.langdetect.profiles.LanguageProfile;
import com.optimaize.langdetect.profiles.LanguageProfileReader;

public class TextUtils {

    private static final LanguageDetector LANG_DETECTOR = createLangDetector();

    public static String extractHtml(ArchiveRecord r) {
        try {
            byte[] rawData = IOUtils.toByteArray(r, r.available());
            String rawContent = new String(rawData, "UTF-8");
            String[] split = rawContent.split("(\r?\n){2}", 2);
            return split[1].trim();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static Optional<String> extractText(String html) {
        Document document = Jsoup.parse(html);
        Element body = document.body();
        if (body == null) {
            return Optional.empty();
        }

        JsoupTextExtractor textExtractor = new JsoupTextExtractor();
        body.traverse(textExtractor);
        String text = textExtractor.getText();
        return Optional.of(text);
    }

    public static String clean(String input) {
        input = StringEscapeUtils.unescapeHtml4(input);
        input = cleanMarkup(input);
        return input;
    }

    public static String cleanMarkup(String line) {
        String after = line.replaceAll("</?\\w+(\\s.+?)?>", " ");
        // .replaceAll("\\[/?\\w+(\\s.+?)?\\]", " ");
        return after;
    }

    public static String languageDetect(String text) {
        com.google.common.base.Optional<LdLocale> result = LANG_DETECTOR.detect(text);

        if (result.isPresent()) {
            return result.get().getLanguage();
        } else {
            return "unk";
        }
    }

    private static LanguageDetector createLangDetector() {
        try {
            List<LanguageProfile> languageProfiles = new LanguageProfileReader().readAllBuiltIn();
            return LanguageDetectorBuilder.create(NgramExtractors.standard())
                    .withProfiles(languageProfiles)
                    .build();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static final Set<String> EN_STOPWORDS = ImmutableSet.of("a", "an", "and", "are", "as", "at", "be",
            "but", "by", "for", "if", "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the",
            "their", "then", "there", "these", "they", "this", "to", "was", "will", "with", "what", "which", "s", "m", "t");

    public static List<String> tokenize(String line) {
        Pattern pattern = Pattern.compile("\\W+");
        String[] split = pattern.split(line.toLowerCase());
        return Arrays.stream(split)
                .map(String::trim)
                .filter(s -> s.length() > 2)
                .collect(Collectors.toList());
    }

    public static List<String> tokenizeFilter(String line) {
        Pattern pattern = Pattern.compile("\\W+");
        String[] split = pattern.split(line.toLowerCase());
        return Arrays.stream(split)
                .map(String::trim)
                .filter(s -> s.length() > 2)
                .filter(s -> !isStopword(s))
                .collect(Collectors.toList());
    }

    public static boolean isStopword(String token) {
        return EN_STOPWORDS.contains(token);
    }

    public static List<String> removeStopwords(List<String> line) {
        return removeStopwords(line, EN_STOPWORDS);
    }

    public static List<String> removeStopwords(List<String> line, Set<String> stopwords) {
        return line.stream().filter(token -> !stopwords.contains(token)).collect(Collectors.toList());
    }
}
