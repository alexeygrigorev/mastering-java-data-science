package chapter02;

import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.LineIterator;
import org.jooq.lambda.tuple.Tuple2;

import com.aol.cyclops.control.LazyReact;
import com.aol.cyclops.types.futurestream.LazyFutureStream;

public class ReactExample {

    public static void main(String[] args) throws IOException {
        LineIterator it = FileUtils.lineIterator(new File("data/words.txt"), "UTF-8");
        ExecutorService executor = Executors.newCachedThreadPool();
        LazyFutureStream<String> stream = 
                LazyReact.parallelBuilder().withExecutor(executor).from(it);

        Map<String, Integer> map = stream
                .map(line -> line.split("\t"))
                .map(arr -> arr[1].toLowerCase())
                .distinct()
                .zipWithIndex()
                .toMap(Tuple2::v1, t -> t.v2.intValue());

        System.out.println(map);
        executor.shutdown();
        it.close();
    }
}
