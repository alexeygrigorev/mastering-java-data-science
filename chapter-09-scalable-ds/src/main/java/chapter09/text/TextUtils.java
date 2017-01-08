package chapter09.text;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import org.apache.commons.io.IOUtils;
import org.archive.io.ArchiveRecord;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;

import com.optimaize.langdetect.LanguageDetector;
import com.optimaize.langdetect.LanguageDetectorBuilder;
import com.optimaize.langdetect.i18n.LdLocale;
import com.optimaize.langdetect.ngram.NgramExtractors;
import com.optimaize.langdetect.profiles.LanguageProfile;
import com.optimaize.langdetect.profiles.LanguageProfileReader;

public class TextUtils {

    private static final Pattern NOT_WHITESPACE_PATTERN = Pattern.compile("\\W+");
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

    public static List<String> tokenize(String line) {
        String[] split = NOT_WHITESPACE_PATTERN.split(line.toLowerCase());
        return Arrays.stream(split)
                .map(String::trim)
                .filter(s -> s.length() > 2)
                .collect(Collectors.toList());
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

}
