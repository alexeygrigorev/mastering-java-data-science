package chapter02.crawl;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;

import org.apache.commons.io.FileUtils;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.jr.ob.JSON;

public class PageFeatureExtractor {

    private static final Logger LOGGER = LoggerFactory.getLogger(PageFeatureExtractor.class);

    public static void main(String[] args) throws IOException {
        try (UrlRepository urls = new UrlRepository()) {
            calculateFeatures(urls);
        }
    }

    private static void calculateFeatures(UrlRepository urls) throws IOException, FileNotFoundException {
        Path path = Paths.get("data/search-results.txt");
        List<String> lines = FileUtils.readLines(path.toFile(), StandardCharsets.UTF_8);

        Stream<RankedPage> pages = lines.parallelStream().flatMap(line -> {
            String[] split = line.split("\t");
            String query = split[0];
            int position = Integer.parseInt(split[1]);
            int searchPageNumber = 1 + (position - 1) / 10;
            String url = "http://" + split[2];

            RankedPage page = new RankedPage(url, position, searchPageNumber);

            Optional<String> html = urls.get(url);
            if (!html.isPresent()) {
                LOGGER.info("page {} for query '{}' wasn't crawled", url, query);
                return Stream.empty();
            }

            Document document = Jsoup.parse(html.get());
            String title = document.title();
            int titleLength = title.length();
            page.setTitleLength(titleLength);

            boolean queryInTitle = title.toLowerCase().contains(query.toLowerCase());
            page.setQueryInTitle(queryInTitle);

            Element body = document.body();
            if (body == null) {
                LOGGER.info("page {} for query '{}' has empty body", url, query);
                return Stream.empty();
            }
            int bodyContentLength = body.text().length();
            page.setBodyContentLength(bodyContentLength);

            int numberOfLinks = document.body().select("a").size();
            page.setNumberOfLinks(numberOfLinks);

            int numberOfHeaders = document.body().select("h1,h2,h3,h4,h5,h6").size();
            page.setNumberOfHeaders(numberOfHeaders);

            return Stream.of(page);
        });

        try (PrintWriter pw = new PrintWriter("ranked-pages.json")) {
            pages.map(p -> toJson(p)).forEach(pw::println);
        }
    }

    private static String toJson(RankedPage page) {
        try {
            return JSON.std.asString(page);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
