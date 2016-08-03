package chapter02;

import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.jr.ob.JSON;
import com.jayway.jsonpath.JsonPath;
import com.jayway.jsonpath.ReadContext;

public class JsonComplexApi {

    public static void main(String[] args) throws Exception {
        String username = "alexeygrigorev";
        String json = UrlUtils.request("https://api.github.com/users/" + username + "/repos");
        System.out.println(json.substring(0, 250));

        @SuppressWarnings("unchecked")
        List<Map<String, ?>> list = (List<Map<String, ?>>) JSON.std.anyFrom(json);
        String name = (String) list.get(0).get("name");
        System.out.println(name);

        ReadContext ctx = JsonPath.parse(json);
        String query = "$..[?(@.language=='Java' && @.stargazers_count > 0)]full_name";
        List<String> javaProjects = ctx.read(query);
        System.out.println(javaProjects);
    }

}
