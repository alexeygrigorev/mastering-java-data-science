package chapter10.search;

public class SearchResult {

    private String url;
    private String title;

    public SearchResult(String url, String title) {
        this.url = url;
        this.title = title;
    }

    public String getUrl() {
        return url;
    }

    public void setUrl(String url) {
        this.url = url;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    @Override
    public String toString() {
        return "SearchResult [url=" + url + ", title=" + title + "]";
    }

}
