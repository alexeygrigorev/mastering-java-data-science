package chapter06.corenlp;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

public class NerGroup {

    public static List<String> groupNer(List<Word> tokens) {
        if (tokens.isEmpty()) {
            return Collections.emptyList();
        }

        String prevNer = "O";
        List<List<Word>> groups = new ArrayList<>();
        List<Word> group = new ArrayList<>();

        for (Word w : tokens) {
            String ner = w.getNer();

            if (prevNer.equals(ner) && !"O".equals(ner)) {
                group.add(w);
                continue;
            }

            groups.add(group);
            group = new ArrayList<>();
            group.add(w);
            prevNer = ner;
        }

        groups.add(group);

        return groups.stream()
                .filter(l -> !l.isEmpty())
                .map(list -> joinLemmas(list))
                .collect(Collectors.toList());
    }

    private static String joinLemmas(List<Word> list) {
        return list.stream()
                .map(Word::getLemma)
                .collect(Collectors.joining(" "));
    }
}
