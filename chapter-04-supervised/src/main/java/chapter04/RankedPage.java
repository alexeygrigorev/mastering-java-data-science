package chapter04;

import com.google.common.base.CharMatcher;

public class RankedPage {

    private static final CharMatcher SLASH = CharMatcher.is('/');

    private String url;
    private int position;
    private int page;
    private int titleLength;
    private int bodyContentLength;
    private boolean queryInTitle;
    private int numberOfHeaders;
    private int numberOfLinks;

    public RankedPage() {
    }

    public RankedPage(String url, int position, int page) {
        this.url = url;
        this.position = position;
        this.page = page;
    }

    public int getUrlLength() {
        return url.length();
    }

    public boolean isHttps() {
        return url.startsWith("https://");
    }

    public boolean isDomainCom() {
        return url.contains(".com");
    }

    public boolean isDomainNet() {
        return url.contains(".net");
    }

    public boolean isDomainOrg() {
        return url.contains(".org");
    }

    public int getNumberOfSlashes() {
        return SLASH.countIn(url);
    }

    public String getUrl() {
        return url;
    }

    public void setUrl(String url) {
        this.url = url;
    }

    public int getPosition() {
        return position;
    }

    public void setPosition(int position) {
        this.position = position;
    }

    public int getPage() {
        return page;
    }

    public void setPage(int page) {
        this.page = page;
    }

    public int getTitleLength() {
        return titleLength;
    }

    public void setTitleLength(int titleLength) {
        this.titleLength = titleLength;
    }

    public int getBodyContentLength() {
        return bodyContentLength;
    }

    public void setBodyContentLength(int bodyContentLength) {
        this.bodyContentLength = bodyContentLength;
    }

    public boolean isQueryInTitle() {
        return queryInTitle;
    }

    public void setQueryInTitle(boolean queryInTitle) {
        this.queryInTitle = queryInTitle;
    }

    public int getNumberOfHeaders() {
        return numberOfHeaders;
    }

    public void setNumberOfHeaders(int numberOfHeaders) {
        this.numberOfHeaders = numberOfHeaders;
    }

    public int getNumberOfLinks() {
        return numberOfLinks;
    }

    public void setNumberOfLinks(int numberOfLinks) {
        this.numberOfLinks = numberOfLinks;
    }

}
