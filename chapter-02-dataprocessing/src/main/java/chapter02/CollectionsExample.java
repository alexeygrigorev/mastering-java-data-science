package chapter02;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class CollectionsExample {

    public static void main(String[] args) {
        List<String> list = new ArrayList<>();
        list.add("alpha");
        list.add("beta");
        list.add("beta");
        list.add("gamma");
        System.out.println(list);

        Set<String> set = new HashSet<>();
        set.add("alpha");
        set.add("beta");
        set.add("beta");
        set.add("gamma");
        System.out.println(set);

        for (String el : set) {
            System.out.println(el);
        }

        Map<String, String> map = new HashMap<>();
        map.put("alpha", "α");
        map.put("beta", "β");
        map.put("gamma", "γ");
        System.out.println(map);

        String min = Collections.min(list);
        String max = Collections.max(list);
        System.out.println("min: " + min + ", max: " + max);
        Collections.sort(list);
        Collections.shuffle(list);

    }

}
