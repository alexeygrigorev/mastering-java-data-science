package chapter02;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.HashMultiset;
import com.google.common.collect.Lists;
import com.google.common.collect.Multiset;
import com.google.common.collect.Multisets;
import com.google.common.collect.Ordering;
import com.google.common.collect.Table;
import com.google.common.io.CharSource;
import com.google.common.io.Files;
import com.google.common.primitives.Ints;

public class GuavaExample {

    public static void main(String[] args) throws IOException {
        File file = new File("data/words.txt");
        CharSource wordsSource = Files.asCharSource(file, StandardCharsets.UTF_8);
        List<String> lines = wordsSource.readLines();

        List<Word> words = Lists.transform(lines, line -> {
            String[] split = line.split("\t");
            return new Word(split[0].toLowerCase(), split[1]);
        });

        Multiset<String> pos = HashMultiset.create();
        for (Word word : words) {
            pos.add(word.getPos());
        }

        Multiset<String> sortedPos = Multisets.copyHighestCountFirst(pos);
        System.out.println(sortedPos);

        ArrayListMultimap<String, String> wordsByPos = ArrayListMultimap.create();
        for (Word word : words) {
            wordsByPos.put(word.getPos(), word.getToken());
        }

        Map<String, Collection<String>> wordsByPosMap = wordsByPos.asMap();
        wordsByPosMap.entrySet().forEach(System.out::println);

        Table<String, String, Integer> table = HashBasedTable.create();
        for (Word word : words) {
            Integer cnt = table.get(word.getPos(), word.getToken());
            if (cnt == null) {
                cnt = 0;
            }
            table.put(word.getPos(), word.getToken(), cnt + 1);
        }

        Map<String, Integer> nouns = table.row("NN");
        System.out.println(nouns);

        Map<String, Integer> posTags = table.column("eu");
        System.out.println(posTags);

        Collection<Integer> values = nouns.values();
        int[] nounCounts = Ints.toArray(values);
        int totalNounCount = Arrays.stream(nounCounts).sum();
        System.out.println(totalNounCount);

        Ordering<Word> byTokenLength = 
                Ordering.natural().<Word> onResultOf(w -> w.getToken().length()).reverse();
        List<Word> sortedByLength = byTokenLength.immutableSortedCopy(words);
        System.out.println(sortedByLength);

        List<Word> sortedCopy = new ArrayList<>(words);
        Collections.sort(sortedCopy, byTokenLength);

        List<Word> first10 = byTokenLength.leastOf(words, 10);
        System.out.println(first10);
        List<Word> last10 = byTokenLength.greatestOf(words, 10);
        System.out.println(last10);

    }
}
