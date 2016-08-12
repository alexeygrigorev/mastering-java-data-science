package chapter02.crawl;

import java.io.File;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.RandomUtils;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Lists;

import chapter02.UrlUtils;

/**
 * DISCLAIMER: this class only serves as an example of how to extract HTML content from the web.
 * Please consult Bing's Terms and Conditions before using it. Use at your own risk. 
 */
public class BingScraper {

    private static final Logger LOGGER = LoggerFactory.getLogger(BingScraper.class);

    private final int minGracePeriod;
    private final int maxGracePeriod;

    public BingScraper(int minGracePeriod, int maxGracePeriod) {
        this.minGracePeriod = minGracePeriod;
        this.maxGracePeriod = maxGracePeriod;
    }

    public List<BingPage> crawl(String query) throws Exception {
        String bingUrl = "https://www.bing.com/search";
        String country = "us";
        String baseUrl = bingUrl + "?cc=" + country + "&q=" + query.toLowerCase().replace(' ', '+');
        List<BingPage> results = Lists.newArrayListWithExpectedSize(35);

        int position = 1;
        for (int page = 0; page < 3; page++) {
            String url = baseUrl + "&first=" + position;
            LOGGER.info("scraping page {} for {}, url: {}", page, query, url);
            String html = UrlUtils.request(url);
            Document document = Jsoup.parse(html);
            Elements searchResults = document.select("ol#b_results li.b_algo");
            for (Element element : searchResults) {
                String link = element.select("h2 a").attr("href");
                results.add(new BingPage(query, page, position, link));
                position++;
            }

            int sleepTime = RandomUtils.nextInt(minGracePeriod, maxGracePeriod);
            LOGGER.info("sleeing {} ms", sleepTime);
            Thread.sleep(sleepTime);
        }

        return deduplicate(results);
    }

    private static List<BingPage> deduplicate(List<BingPage> results) {
        Set<String> seen = new HashSet<>();
        List<BingPage> dedup = Lists.newArrayListWithExpectedSize(35);

        int duplicatesCount = 0;
        for (BingPage page : results) {
            if (seen.contains(page.getUrl())) {
                duplicatesCount++;
                continue;
            }
            seen.add(page.getUrl());
            dedup.add(page);
        }

        LOGGER.info("removed {} duplicates", duplicatesCount);
        return dedup;
    }

    public static class BingPage {
        private final String query;
        private final int page;
        private final int position;
        private final String url;

        public BingPage(String query, int page, int position, String url) {
            this.query = query;
            this.page = page;
            this.position = position;
            this.url = url;
        }

        public String getQuery() {
            return query;
        }

        public int getPage() {
            return page;
        }

        public int getPosition() {
            return position;
        }

        public String getUrl() {
            return url;
        }

        @Override
        public String toString() {
            return "BingPage [query=" + query + ", position=" + position + ", url=" + url + "]";
        }

    }

    public static void main(String[] args) throws Exception {
        BingScraper bingCrawler = new BingScraper(500, 2000);

        List<String> queries = FileUtils.readLines(new File("data/keywords.txt"), StandardCharsets.UTF_8);
        try (PrintWriter pw = new PrintWriter("bing-search-results.txt")) {
            for (String query : queries) {
                LOGGER.info("crawing {}", query);
                List<BingPage> crawl = bingCrawler.crawl(query);
                for (BingPage page : crawl) {
                    pw.print(page.getQuery());
                    pw.print('\t');
                    pw.print(page.getPage());
                    pw.print('\t');
                    pw.print(page.getPosition());
                    pw.print('\t');
                    pw.print(page.getUrl());
                    pw.println();
                }
                pw.flush();
            }
        }
    }
}
