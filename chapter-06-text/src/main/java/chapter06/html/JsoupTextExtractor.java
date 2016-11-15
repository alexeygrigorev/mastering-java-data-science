package chapter06.html;

import java.util.Arrays;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import org.apache.commons.lang3.StringUtils;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.Node;
import org.jsoup.nodes.TextNode;
import org.jsoup.select.NodeVisitor;

import com.google.common.collect.ImmutableSet;

public class JsoupTextExtractor implements NodeVisitor {
    public static final String BLOCK_SEPARATOR = "[::new_line::]";
    public static final Pattern BLOCK_SEPARATOR_PATTERN = Pattern.compile(Pattern.quote(BLOCK_SEPARATOR));

    private static final Pattern WHITESPACE = Pattern.compile("[\u00A0\\s]+", Pattern.DOTALL);

    public static final Set<String> BLOCK_TAGS = ImmutableSet.of("h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "ol",
            "br", "hr", "tr", "td", "div", "pre", "option");

    private final StringBuilder allText = new StringBuilder(100);

    @Override
    public void head(Node node, int depth) {
        if (node instanceof TextNode) {
            TextNode textNode = (TextNode) node;
            String text = textNode.getWholeText();
            text = text.trim();
            if (text.isEmpty()) {
                return;
            }

            text = WHITESPACE.matcher(text).replaceAll(" ").trim();
            if (text.isEmpty()) {
                return;
            }

            text = UnicodeUtils.fixUnicode(text);
            allText.append(text).append(" ");
        }
    }

    @Override
    public void tail(Node node, int depth) {
        if (node instanceof Element) {
            Element element = (Element) node;
            String tagName = element.tagName().toLowerCase();
            if (!BLOCK_TAGS.contains(tagName)) {
                return;
            }
            allText.append(BLOCK_SEPARATOR);
        }
    }


    public String getText() {
        return getText("\n");
    }

    public String getText(String newLineSeparator) {
        String text = allText.toString();
        String[] split = BLOCK_SEPARATOR_PATTERN.split(text);
        return Arrays.stream(split)
                .map(String::trim)
                .filter(StringUtils::isNotEmpty)
                .collect(Collectors.joining(newLineSeparator));
    }
}
