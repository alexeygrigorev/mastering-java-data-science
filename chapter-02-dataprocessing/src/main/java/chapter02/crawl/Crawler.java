package chapter02.crawl;

import java.io.IOException;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import chapter02.UrlUtils;

public class Crawler implements AutoCloseable {

    private static final Logger LOGGER = LoggerFactory.getLogger(Crawler.class);

    private final ExecutorService executor;
    private final int timeout;

    public Crawler(int timeout) {
        this.executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        this.timeout = timeout;
    }

    public Optional<String> crawl(String url) throws IOException {
        try {
            Future<String> future = executor.submit(() -> UrlUtils.request(url));
            String result = future.get(timeout, TimeUnit.SECONDS);
            if (!result.isEmpty()) {
                return Optional.of(result);
            } else {
                LOGGER.info("crawled empty result for {}", url);
                return Optional.empty();
            }
        } catch (TimeoutException e) {
            LOGGER.warn("timeout exception: could not crawl {} in {} sec", url, timeout);
            return Optional.empty();
        } catch (Exception e) {
            LOGGER.error("something happened during crawling", e);
            return Optional.empty();
        }
    }

    @Override
    public void close() {
        executor.shutdown();
    }

}
