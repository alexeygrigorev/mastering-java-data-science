package chapter02;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class StreamsExample {

    public static void main(String[] args) throws IOException {
        Word[] array = { new Word("My", "RPR"), new Word("dog", "NN"), new Word("also", "RB"), 
                new Word("likes", "VB"), new Word("eating", "VB"), new Word("sausage", "NN"), 
                new Word(".", ".") };
        List<Word> list = Arrays.asList(array);

        List<String> nounts = list.stream()
                .filter(w -> "NN".equals(w.getPos()))
                .map(Word::getToken)
                .collect(Collectors.toList());
        System.out.println(nounts);

        Set<String> pos = list.stream()
                .map(Word::getPos)
                .collect(Collectors.toSet());
        System.out.println(pos);

        String rawSentence = list.stream()
                .map(Word::getToken)
                .collect(Collectors.joining(" "));
        System.out.println(rawSentence);

        Map<String, List<Word>> groupByPos = list.stream()
                .collect(Collectors.groupingBy(Word::getPos));
        System.out.println(groupByPos.get("VB"));
        System.out.println(groupByPos.get("NN"));

        Map<String, Word> tokenToWord = list.stream()
                .collect(Collectors.toMap(Word::getToken, Function.identity()));
        System.out.println(tokenToWord.get("sausage"));

        int maxTokenLength = list.stream()
                .mapToInt(w -> w.getToken().length())
                .max().getAsInt();
        System.out.println(maxTokenLength);

        int[] firstLengths = list.parallelStream()
                .filter(w -> w.getToken().length() % 2 == 0)
                .map(Word::getToken)
                .mapToInt(String::length)
                .sequential()
                .sorted()
                .limit(2)
                .toArray();
        System.out.println(Arrays.toString(firstLengths));

        Path path = Paths.get("text.txt");
        try (Stream<String> lines = Files.lines(path, StandardCharsets.UTF_8)) {
            double average = lines
                .flatMap(line -> Arrays.stream(line.split(" ")))
                .map(String::toLowerCase)
                .mapToInt(String::length)
                .average().getAsDouble();
            System.out.println("average token length: " + average);
        }
    }

}
