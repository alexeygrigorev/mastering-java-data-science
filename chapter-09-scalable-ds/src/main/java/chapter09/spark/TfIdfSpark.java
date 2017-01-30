package chapter09.spark;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import scala.Tuple2;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;
import com.google.common.collect.Multiset.Entry;
import com.google.common.collect.Sets;

public class TfIdfSpark {

    private static final double LOG_N = Math.log(1_000_000);

    public static void main(String[] args) {
        System.setProperty("hadoop.home.dir", "c:/tmp/hadoop");

        SparkConf conf = new SparkConf().setAppName("tfidf").setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);

        JavaRDD<String> textFile = sc.textFile("C:/tmp/warc");
        textFile.take(10).forEach(System.out::println);

        JavaPairRDD<String, Integer> dfRdd = textFile
                .flatMap(line ->  distinctTokens(line))
                .mapToPair(t -> new Tuple2<>(t, 1))
                .reduceByKey((a, b) -> a + b)
                .filter(t -> t._2 >= 100);

        dfRdd.take(10).forEach(System.out::println);
        Map<String, Integer> dfs = dfRdd.collectAsMap();

        JavaRDD<String> tfIdfRdd = textFile.map(line -> tfIdfDocument(dfs, line));

        tfIdfRdd.take(10).forEach(System.out::println);
        tfIdfRdd.saveAsTextFile("c:/tmp/warc-tfidf");

        sc.close();
    }

    private static String tfIdfDocument(Map<String, Integer> dfs, String line) {
        String[] split = line.split("\t");
        String url = split[0];
        List<String> tokens = Arrays.asList(split[1].split(" "));
        Multiset<String> counts = HashMultiset.create(tokens);

        String tfIdfTokens = counts.entrySet().stream()
                .map(e -> toTfIdf(dfs, e))
                .collect(Collectors.joining(" "));

        return url + "\t" + tfIdfTokens;
    }

    private static String toTfIdf(Map<String, Integer> dfs, Entry<String> e) {
        String token = e.getElement();
        int tf = e.getCount();

        int df = dfs.getOrDefault(token, 100);

        double idf = LOG_N - Math.log(df);
        double tfIdf = tf * idf;

        return String.format("%s:%.5f", token, tfIdf);
    }

    private static Iterator<String> distinctTokens(String line) {
        String[] split = line.split("\t");
        Set<String> tokens = Sets.newHashSet(split[1].split(" "));
        return tokens.iterator();
    }

}
