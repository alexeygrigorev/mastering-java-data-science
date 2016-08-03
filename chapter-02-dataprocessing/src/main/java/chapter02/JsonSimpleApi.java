package chapter02;

import java.util.Map;

import com.fasterxml.jackson.jr.ob.JSON;

public class JsonSimpleApi {

    public static void main(String[] args) throws Exception {
        String text = "mastering java for data science";
        String json = UrlUtils.request("http://md5.jsontest.com/?text=" + text.replace(' ', '+'));

        Map<String, Object> map = JSON.std.mapFrom(json);
        System.out.println(map.get("original"));
        System.out.println(map.get("md5"));
    }

}
