package chapter06.ml;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.lang3.StringUtils;

import chapter06.text.CountVectorizer;
import chapter06.text.TextUtils;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.SolverType;
import smile.data.SparseDataset;
import smile.nlp.stemmer.PorterStemmer;

public class Sentiment {

    private static final Pattern HTML = Pattern.compile("<.+?>");

    public static void main(String[] args) throws IOException {
        List<SentimentRecord> data = readFromTagGz("data/aclImdb_v1.tar.gz");

        List<SentimentRecord> train = data.stream().filter(SentimentRecord::isTrain).collect(Collectors.toList());

        List<List<String>> trainTokens = tokenizeSentimentContent(train);

        CountVectorizer cv = CountVectorizer.create()
                .withIdfTransformation()
                .withL2Normalization()
                .withMinimalDocumentFrequency(10)
                .withSublinearTfTransformation()
                .build();

        SparseDataset trainData = cv.fitTransform(trainTokens);
        double[] y = labels(train);

        SparseLibLinear.mute();
        Parameter param = new Parameter(SolverType.L1R_LR, 1, 0.00001);
        Model model = SparseLibLinear.train(trainData, y, param);

        List<SentimentRecord> test = data.stream().filter(SentimentRecord::isTest).collect(Collectors.toList());
        List<List<String>> testTokens = tokenizeSentimentContent(test);
        SparseDataset testData = cv.transfrom(testTokens);

        double[] proba = SparseLibLinear.predictProba(model, testData);
        double[] actual = labels(test);
        double auc = Metrics.auc(actual, proba);

        System.out.println(auc);
    }


    private static double[] labels(List<SentimentRecord> data) {
        return data.stream().mapToDouble(s -> s.getLabel() ? 1.0 : 0.0).toArray();
    }


    private static List<List<String>> tokenizeSentimentContent(List<SentimentRecord> train) {
        Stream<List<String>> tokenStream = train.stream()
                .map(SentimentRecord::getContent)
                .map(Sentiment::tokenize);
        return tokenStream.collect(Collectors.toList());
    }

    private static List<SentimentRecord> readFromTagGz(String path) throws IOException {
        List<SentimentRecord> data = new ArrayList<>();
        Path archive = Paths.get(path);
        try (InputStream is = Files.newInputStream(archive);
                BufferedInputStream bis = new BufferedInputStream(is);
                GzipCompressorInputStream gis = new GzipCompressorInputStream(bis);
                TarArchiveInputStream tar = new TarArchiveInputStream(gis)) {

            while (true) {
                TarArchiveEntry nextTarEntry = tar.getNextTarEntry();
                if (nextTarEntry == null) {
                    break;
                }

                String name = nextTarEntry.getName();
                if (!name.endsWith(".txt")) {
                    continue;
                }

                String[] split = name.split("/");
                if (split.length != 4) {
                    continue;
                }

                String label = split[2];
                if ("unsup".equals(label)) {
                    continue;
                }

                String id = StringUtils.removeEnd(split[3], ".txt");
                boolean boolLabel = "pos".equals(label);
                boolean train = "train".equals(split[1]);

                int recordSize = tar.getRecordSize();
                byte[] buff = new byte[recordSize];
                tar.read(buff);

                String content = new String(buff, StandardCharsets.UTF_8);
                data.add(new SentimentRecord(id, train, boolLabel, content));
            }
        }

        return data;
    }

    public static List<String> tokenize(String line) {
        PorterStemmer stemmer = new PorterStemmer();

        line = stripHtml(line);
        Pattern pattern = Pattern.compile("\\W+");
        String[] split = pattern.split(line.toLowerCase());
        return Arrays.stream(split)
                .map(String::trim)
                .filter(s -> s.length() > 2)
                .filter(s -> !TextUtils.isStopword(s))
                .map(stemmer::stem)
                .collect(Collectors.toList());
    }

    
    private static String stripHtml(String content) {
        return HTML.matcher(content).replaceAll(" ");
    }

    private static class SentimentRecord {
        private final String id;
        private final boolean train;
        private final boolean label;
        private final String content;

        public SentimentRecord(String id, boolean train, boolean label, String content) {
            this.id = id;
            this.train = train;
            this.label = label;
            this.content = content;
        }

        @Override
        public String toString() {
            return "SentimentRecord [id=" + id + ", train=" + train + ", label=" + label + ", content=" + content + "]";
        }

        public boolean getLabel() {
            return label;
        }

        public boolean isTrain() {
            return train;
        }

        public boolean isTest() {
            return !train;
        }

        public String getContent() {
            return content;
        }

    }
}
