package chapter02;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;

public class CommonsIOExample {

    public static void main(String[] args) throws IOException {
        fileUtils();
        ioUtils();
    }

    private static void fileUtils() throws IOException {
        File textFile = new File("data/text.txt");
        String content = FileUtils.readFileToString(textFile, StandardCharsets.UTF_8);
        System.out.println(content);
        List<String> lines = FileUtils.readLines(textFile, StandardCharsets.UTF_8);
        System.out.println(lines);
    }

    private static void ioUtils() throws IOException, FileNotFoundException {
        try (InputStream is = new FileInputStream("data/text.txt")) {
            String content = IOUtils.toString(is, StandardCharsets.UTF_8);
            System.out.println(content);
        }

        try (InputStream is = new FileInputStream("data/text.txt")) {
            List<String> lines = IOUtils.readLines(is, StandardCharsets.UTF_8);
            System.out.println(lines);
        }
    }
}
