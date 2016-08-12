package chapter02;

import java.io.IOException;
import java.io.InputStream;
import java.io.UnsupportedEncodingException;
import java.net.URL;
import java.net.URLConnection;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;

import org.apache.commons.io.IOUtils;

public class UrlUtils {
    public static String request(String url) throws IOException {
        try (InputStream is = new URL(url).openStream()) {
            return IOUtils.toString(is, StandardCharsets.UTF_8);
        }
    }

    public static String userAgentRequest(String address) throws IOException {
        URL url = new URL(address);
        URLConnection con = url.openConnection();
        con.setRequestProperty("User-Agent",
                "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0");
        con.setRequestProperty("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8");
        con.setRequestProperty("Accept-Language", "en-US,en;q=0.5");

        try (InputStream is = con.getInputStream()) {
            return IOUtils.toString(is, StandardCharsets.UTF_8);
        }
    }

    public static String encode(Object in) {
        try {
            return URLEncoder.encode(String.valueOf(in), "UTF-8");
        } catch (UnsupportedEncodingException e) {
            throw new RuntimeException(e);
        }
    }
}
