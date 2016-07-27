package chapter02.html;

public class RankedPage {

    private final String query;
    private final int rank;
    private final String url;
    private final String html;

    public RankedPage(String query, int rank, String url, String html) {
        this.query = query;
        this.rank = rank;
        this.url = url;
        this.html = html;
    }

    public String getQuery() {
        return query;
    }

    public int getRank() {
        return rank;
    }

    public String getUrl() {
        return url;
    }

    public String getHtml() {
        return html;
    }

}
