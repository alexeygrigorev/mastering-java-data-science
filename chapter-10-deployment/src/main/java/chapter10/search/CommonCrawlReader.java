package chapter10.search;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.Collections;
import java.util.Iterator;
import java.util.Optional;

import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.StringUtils;
import org.archive.io.ArchiveReader;
import org.archive.io.ArchiveRecord;
import org.archive.io.warc.WARCReaderFactory;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import com.google.common.collect.AbstractIterator;
import com.google.common.collect.ArrayListMultimap;

import chapter10.html.JsoupTextExtractor;
import chapter10.html.TextUtils;

public class CommonCrawlReader {

    public static Iterator<HtmlDocument> iterator(File commonCrawlFile) throws IOException {
        try {
            ArchiveReader archive = WARCReaderFactory.get(commonCrawlFile);
            Iterator<ArchiveRecord> records = archive.iterator();

            return new AbstractIterator<HtmlDocument>() {
                @Override
                protected HtmlDocument computeNext() {
                    while (records.hasNext()) {
                        try {
                            ArchiveRecord record = records.next();
                            Optional<HtmlDocument> extracted = extractText(record);
                            if (extracted.isPresent()) {
                                return extracted.get();
                            }
                        } catch (Exception e) {
                            // ignoring it
                        }
                    }

                    IOUtils.closeQuietly(archive);
                    return endOfData();
                }
            };
        } catch (Exception e) {
            return Collections.emptyIterator();
        }
    }

    private static Optional<HtmlDocument> extractText(ArchiveRecord record) {
        String url = record.getHeader().getUrl();
        if (StringUtils.isBlank(url)) {
            // if there's no URL associated with a page, it's not a web page
            return Optional.empty();
        }

        String html = TextUtils.extractHtml(record);
        Document document = Jsoup.parse(html);
        String title = document.title();
        if (title == null) {
            return Optional.empty();
        }

        Element body = document.body();
        if (body == null) {
            return Optional.empty();
        }

        JsoupTextExtractor textExtractor = new JsoupTextExtractor();
        body.traverse(textExtractor);
        String bodyText = textExtractor.getText();

        Elements headerElements = body.select("h1, h2, h3, h4, h5, h6");
        ArrayListMultimap<String, String> headers = ArrayListMultimap.create();
        for (Element htag : headerElements) {
            String tagName = htag.nodeName().toLowerCase();
            headers.put(tagName, htag.text());
        }

        return Optional.of(new HtmlDocument(url, title, headers, bodyText));
    }

    public static class HtmlDocument implements Serializable {
        private final String url;
        private final String title;
        private final ArrayListMultimap<String, String> headers;
        private final String bodyText;

        public HtmlDocument(String url, String title, ArrayListMultimap<String, String> headers, String bodyText) {
            this.url = url;
            this.title = title;
            this.headers = headers;
            this.bodyText = bodyText;
        }

        public String getUrl() {
            return url;
        }

        public String getTitle() {
            return title;
        }

        public ArrayListMultimap<String, String> getHeaders() {
            return headers;
        }

        public String getBodyText() {
            return bodyText;
        }
    }

}
