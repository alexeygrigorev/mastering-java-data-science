package chapter02;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.io.IOUtils;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

public class JsoupExample {

    public static void main(String[] args) throws IOException {
        Map<String, Double> result = new HashMap<>();

        String rawHtml = html("https://www.kaggle.com/c/avito-duplicate-ads-detection/leaderboard");
        Document document = Jsoup.parse(rawHtml);
        Elements tableRows = document.select("#leaderboard-table tr");
        for (Element tr : tableRows) {
            Elements columns = tr.select("td");
            if (columns.isEmpty()) {
                continue;
            }

            String team = columns.get(2).select("a.team-link").text();
            double score = Double.parseDouble(columns.get(3).text());
            result.put(team, score);
        }

        Comparator<Map.Entry<String, Double>> byValue = Map.Entry.comparingByValue();
        result.entrySet().stream()
                .sorted(byValue.reversed())
                .forEach(System.out::println);
    }

    public static String html(String url) throws IOException {
        try (InputStream is = new URL(url).openStream()) {
            return IOUtils.toString(is, StandardCharsets.UTF_8);
        }
    }

}
