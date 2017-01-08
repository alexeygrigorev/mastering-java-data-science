package chapter09.hadoop;

import java.io.IOException;
import java.util.List;
import java.util.Optional;

import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.archive.io.ArchiveReader;
import org.archive.io.ArchiveRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import chapter09.text.TextUtils;

public class WarcPreparationMapper extends Mapper<Text, ArchiveReader, Text, NullWritable> {

    private static final Logger LOGGER = LoggerFactory.getLogger(WarcPreparationMapper.class);

    private static final int MAX_DOCUMENT_LEN = 5000000;
    private static final int MIN_DOCUMENT_LEN = 100;

    private Text output = new Text();
    private long documentcounter;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        documentcounter = 0;
    }

    @Override
    protected void map(Text input, ArchiveReader archive, Context context) throws IOException, InterruptedException {
        try {
            for (ArchiveRecord record : archive) {
                try {
                    process(record, context);

                    if (documentcounter % 1000 == 0) {
                        LOGGER.info("processed {} documents so far", documentcounter);
                    }
                } catch (Exception ex) {
                    LOGGER.error("Caught Exception", ex);
                }
            }
        } catch (Exception ex) {
            LOGGER.error("Caught Exception", ex);
        }
    }

    private void process(ArchiveRecord record, Context context) throws Exception {
        String url = record.getHeader().getUrl();
        if (StringUtils.isBlank(url)) {
            // if there's no URL associated with a page, it's not a web page
            return;
        }

        int documentLength = record.available();
        if (documentLength > MAX_DOCUMENT_LEN) {
            LOGGER.info("the document at {} is too big ({} bytes). Skipping it", url, documentLength);
            return;
        }

        String html = TextUtils.extractHtml(record);
        if (html.length() <= MIN_DOCUMENT_LEN) {
            return;
        }

        Optional<String> text = TextUtils.extractText(html);
        if (!text.isPresent()) {
            return;
        }

        String lang = TextUtils.languageDetect(text.get());
        if (!lang.equals("en")) {
            return;
        }

        List<String> tokens = TextUtils.tokenize(text.get());
        if (tokens.isEmpty()) {
            return;
        }

        String result = url + "\t" + String.join(" ", tokens);
        output.set(result);

        context.write(output, NullWritable.get());
    }

    @Override
    protected void cleanup(Mapper<Text, ArchiveReader, Text, NullWritable>.Context context)
            throws IOException, InterruptedException {
        LOGGER.info("done");
    }

}
