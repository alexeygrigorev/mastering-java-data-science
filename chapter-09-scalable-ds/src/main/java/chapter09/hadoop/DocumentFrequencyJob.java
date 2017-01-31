package chapter09.hadoop;

import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.compress.GzipCodec;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DocumentFrequencyJob extends Configured implements Tool {

    private static final Logger LOGGER = LoggerFactory.getLogger(DocumentFrequencyJob.class);

    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            args = new String[] { 
                "--input", "/home/agrigorev/Downloads/cc/warc-processed", 
                "--output", "/home/agrigorev/Downloads/cc/warc-df" 
            };
        }

        int res = ToolRunner.run(new Configuration(), new DocumentFrequencyJob(), args);
        System.exit(res);
    }

    public int run(String[] args) throws Exception {
        Map<String, String> params = Params.parse(args);
        LOGGER.info("using paramgs: {}", params);

        Job job = Job.getInstance(getConf());

        job.setJobName("DF job");
        job.setJarByClass(this.getClass());

        Path inputPath = new Path(params.get("--input"));
        Path outputPath = new Path(params.get("--output"));

        LOGGER.info("Input path: {}", inputPath);
        FileInputFormat.addInputPath(job, inputPath);

        TextOutputFormat.setOutputPath(job, outputPath);
        TextOutputFormat.setCompressOutput(job, true);
        TextOutputFormat.setOutputCompressorClass(job, GzipCodec.class);

        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(LongWritable.class);

        job.setMapperClass(DocumentFrequencyMapper.class);
        job.setCombinerClass(DocumentFrequencyReducer.class);
        job.setReducerClass(DocumentFrequencyReducer.class);

        if (job.waitForCompletion(true)) {
            return 0;
        } else {
            return 1;
        }
    }
}
