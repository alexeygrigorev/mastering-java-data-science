package chapter09.hadoop;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class DocumentFrequencyMapper extends Mapper<LongWritable, Text, Text, LongWritable> {

    private Text outToken = new Text();
    private LongWritable one = new LongWritable(1);

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String doc = value.toString();
        String[] split = doc.split("\t");

        String joinedTokens = split[1];
        List<String> tokens = Arrays.asList(joinedTokens.split(" "));

        Set<String> distinct = new HashSet<>(tokens); 

        for (String token : distinct) {
            outToken.set(token);
            context.write(outToken, one);
        }
    }
}
