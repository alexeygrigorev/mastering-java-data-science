package chapter09.hadoop;

import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class DocumentFrequencyReducer extends Reducer<Text, LongWritable, Text, LongWritable> {

    private LongWritable out = new LongWritable();

    @Override
    protected void reduce(Text key, Iterable<LongWritable> values, Context context)
            throws IOException, InterruptedException {
        long sum = 0;
        for (LongWritable cnt : values) {
            sum = sum + cnt.get();
        }

        if (sum > 100) {
            out.set(sum);
            context.write(key, out);
        }
    }
}
