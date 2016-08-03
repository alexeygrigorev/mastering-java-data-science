package chapter02.crawl;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Optional;

import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CrawlerExample {

    private static final Logger LOGGER = LoggerFactory.getLogger(CrawlerExample.class);

    public static void main(String[] args) throws IOException {
        try (Crawler crawler = new Crawler(10)) {
            try (UrlRepository urls = new UrlRepository()) {
                crawl(crawler, urls);
            }
        }
    }

    private static void crawl(Crawler crawler, UrlRepository urls) throws IOException {
        Path path = Paths.get("data/search-results.txt");
        List<String> lines = FileUtils.readLines(path.toFile(), StandardCharsets.UTF_8);

        lines.parallelStream()
            .map(line -> line.split("\t"))
            .map(split -> "http://" + split[2])
            .distinct()
            .filter(url -> !urls.contains(url))
            .forEach(url -> {
                try {
                    Optional<String> html = crawler.crawl(url);
                    if (html.isPresent()) {
                        LOGGER.debug("successfully crawled {}", url);
                        urls.put(url, html.get());
                    }
                } catch (Exception e) {
                    LOGGER.error("got exception when processing url {}", url, e);
                }
            });

        LOGGER.info("done");
    }

}
