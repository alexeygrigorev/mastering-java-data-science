package chapter09.hadoop;

import java.util.Map;

import com.google.common.collect.ImmutableMap;

public class Params {

    public static Map<String, String> parse(String[] args) {
        if (args == null || args.length % 2 != 0) {
            throw new IllegalStateException("Cannot convert args!");
        }

        ImmutableMap.Builder<String, String> builder = ImmutableMap.builder();
        for (int n = 0; n < args.length; n += 2) {
            builder.put(args[n], args[n + 1]);
        }

        return builder.build();
    }
}