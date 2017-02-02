package chapter09.hadoop;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;
import com.google.common.collect.Multiset.Entry;

public class IdfMapper extends Mapper<LongWritable, Text, Text, NullWritable> {

    private static final double LOG_N = Math.log(1_000_000);

    private Text output = new Text();

    private Map<String, Integer> dfs;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        dfs = new HashMap<>();

        File dir = new File("./df");
        for (File file : dir.listFiles()) {
            if (file.getName().equals("_SUCCESS")) {
                continue;
            }

            try (FileInputStream is = FileUtils.openInputStream(file)) {
                LineIterator lines = IOUtils.lineIterator(is, StandardCharsets.UTF_8);
                while (lines.hasNext()) {
                    String line = lines.next();
                    String[] split = line.split("\t");
                    if (split.length < 2) {
                        continue;
                    }

                    dfs.put(split[0], Integer.parseInt(split[1]));
                }
            }
        }
    }

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String doc = value.toString();
        String[] split = doc.split("\t");
        String url = split[0];

        List<String> tokens = Arrays.asList(split[1].split(" "));
        Multiset<String> counts = HashMultiset.create(tokens);

        String tfIdfTokens = counts.entrySet().stream()
                .map(e -> toTfIdf(dfs, e))
                .collect(Collectors.joining(" "));

        output.set(url + "\t" + tfIdfTokens);
        context.write(output, NullWritable.get());
    }

    private static String toTfIdf(Map<String, Integer> dfs, Entry<String> e) {
        String token = e.getElement();
        int tf = e.getCount();

        int df = dfs.getOrDefault(token, 100);

        double idf = LOG_N - Math.log(df);
        double tfIdf = tf * idf;

        return String.format("%s:%.5f", token, tfIdf);
    }
}
