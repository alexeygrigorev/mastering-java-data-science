package chapter06.corenlp;

import static org.junit.Assert.*;

import java.util.Arrays;
import java.util.List;

import org.junit.Test;

public class NerGroupTest {

    @Test
    public void test() {
        // 
        List<Word> tokens = Arrays.asList(
                new Word("My", "My", "_", "O"),
                new Word("name", "name", "_", "O"),
                new Word("is", "is", "_", "O"),
                new Word("Justin", "Justin", "_", "PERSON"),
                new Word("Bieber", "Bieber", "_", "PERSON"),
                new Word("I", "I", "_", "O"),
                new Word("live", "live", "_", "O"),
                new Word("in", "in", "_", "O"),
                new Word("Brooklyn", "Brooklyn", "_", "LOCATION"),
                new Word(",", ",", "_", "O"),
                new Word("New", "New", "_", "LOCATION"),
                new Word("York", "York", "_", "LOCATION"),
                new Word(".", ".", "_", "O")
        );

        List<String> results = NerGroup.groupNer(tokens);
        List<String> expected = Arrays.asList("My", "name", "is", "Justin Bieber", "I", "live", "in", "Brooklyn", ",",
                "New York", ".");
        assertEquals(expected, results);
    }

}
