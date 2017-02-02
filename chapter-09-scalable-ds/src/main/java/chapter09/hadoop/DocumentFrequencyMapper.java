package chapter09.hadoop;

import java.io.IOException;
import java.util.Set;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import com.google.common.collect.Sets;

public class DocumentFrequencyMapper extends Mapper<LongWritable, Text, Text, LongWritable> {

    public static enum Counter {
        DOCUMENTS;
    }

    private Text outToken = new Text();
    private LongWritable one = new LongWritable(1);

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String doc = value.toString();
        String[] split = doc.split("\t");

        String joinedTokens = split[1];
        Set<String> tokens = Sets.newHashSet(joinedTokens.split(" "));

        for (String token : tokens) {
            outToken.set(token);
            context.write(outToken, one);
        }

        context.getCounter(Counter.DOCUMENTS).increment(1);
    }
}
