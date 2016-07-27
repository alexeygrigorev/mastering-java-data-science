package chapter02;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.charset.StandardCharsets;

import org.apache.commons.io.IOUtils;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.select.Elements;

public class UrlJsopExample {

    public static void main(String[] args) throws IOException {
        String rawHtml = html("https://www.kaggle.com/c/avito-duplicate-ads-detection/leaderboard");
        Document document = Jsoup.parse(rawHtml);
        Elements tableRows = document.select("#leaderboard-table tr");
        tableRows.forEach(System.out::println);
    }

    public static String html(String url) throws IOException {
        try (InputStream is = new URL(url).openStream()) {
            return IOUtils.toString(is, StandardCharsets.UTF_8);
        }
    }

}
