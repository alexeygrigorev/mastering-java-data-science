package chapter07;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.mapdb.DB;
import org.mapdb.DBMaker;
import org.mapdb.HTreeMap;

public class UrlRepository implements AutoCloseable {

    private final DB db;
    private final Map<String, String> map;

    public UrlRepository() {
        this.db = makeDb();
        this.map = createUrlMapDatabase(this.db);
    }

    public void put(String url, String html) {
        map.put(url, html);
    }

    public boolean contains(String url) {
        return map.containsKey(url);
    }

    public Optional<String> get(String url) {
        if (map.containsKey(url)) {
            return Optional.of(map.get(url));
        } else {
            return Optional.empty();
        }
    }

    public List<String> allUrls() {
        return new ArrayList<>(map.keySet());
    }

    private static DB makeDb() {
        return DBMaker.fileDB("data/urls.db").closeOnJvmShutdown().make();
    }

    private static Map<String, String> createUrlMapDatabase(DB db) {
        HTreeMap<?, ?> htreeMap = db.hashMap("urls").createOrOpen();
        @SuppressWarnings("unchecked")
        Map<String, String> map = (Map<String, String>) htreeMap;
        return map;
    }

    @Override
    public void close() {
        db.close();
    }

}
