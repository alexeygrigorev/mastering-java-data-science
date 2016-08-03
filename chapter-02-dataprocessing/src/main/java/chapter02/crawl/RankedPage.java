package chapter02.crawl;

public class RankedPage {

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
