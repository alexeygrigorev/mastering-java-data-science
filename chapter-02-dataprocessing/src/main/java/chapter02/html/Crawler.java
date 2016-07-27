package chapter02.html;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.stream.Stream;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.mapdb.DB;
import org.mapdb.DBMaker;
import org.mapdb.HTreeMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Crawler {

    private static final Logger LOGGER = LoggerFactory.getLogger(Crawler.class);

    private final Map<String, String> cache;
    private final ExecutorService executor;
    private final int timeout;

    public Crawler(int timeout) {
        this.cache = persistentCache();
        this.executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        this.timeout = timeout;
    }

    public Optional<String> crawl(String url) throws IOException {
        if (cache.containsKey(url)) {
            LOGGER.debug("catche hit for {}", url);
            String result = cache.get(url);
            return Optional.of(result);
        }

        LOGGER.debug("donwloading {}...", url);
        Optional<String> html = doTimeoutedCrawl(url);
        if (html.isPresent()) {
            cache.put(url, html.get());
        }
        return html;
    }

    private Optional<String> doTimeoutedCrawl(String url) {
        try {
            Future<String> future = executor.submit(() -> Crawler.doCrawl(url));
            String result = future.get(timeout, TimeUnit.SECONDS);
            return Optional.of(result);
        } catch (TimeoutException e) {
            LOGGER.warn("timeout exception: could not crawl {} in {} sec", url, timeout);
            return Optional.empty();
        } catch (Exception e) {
            LOGGER.error("something happened during crawling", e);
            return Optional.empty();
        }
    }

    private static String doCrawl(String url) throws IOException {
        try (InputStream is = new URL(url).openStream()) {
            return IOUtils.toString(is, StandardCharsets.UTF_8);
        }
    }

    private static Map<String, String> persistentCache() {
        DB db = DBMaker.fileDB("cache.db").closeOnJvmShutdown().closeOnJvmShutdown().make();
        HTreeMap<?, ?> map = db.hashMap("crawl").expireAfterCreate(2, TimeUnit.DAYS).createOrOpen();
        @SuppressWarnings("unchecked")
        Map<String, String> result = (Map<String, String>) map;
        return result;
    }

    public static void main(String[] args) throws IOException {
        Path path = Paths.get("data/search-results.txt");
        List<String> lines = FileUtils.readLines(path.toFile(), StandardCharsets.UTF_8);

        Crawler crawler = new Crawler(10);

        lines.parallelStream().flatMap(line -> {
            String[] split = line.split("\t");
            String query = split[0];
            int rank = Integer.parseInt(split[1]);
            String url = split[2];

            try {
                Optional<String> html = crawler.crawl("http://" + url);
                if (html.isPresent()) {
                    RankedPage page = new RankedPage(query, rank, url, html.get());
                    return Stream.of(page);
                } else {
                    LOGGER.warn("no html extracted from {}", url);
                }
            } catch (Exception e) {
                LOGGER.error("got exception when processing url {}", url, e);
            }
            return Stream.empty();
        }).map(RankedPage::getUrl).forEach(System.out::println);

    }

}
