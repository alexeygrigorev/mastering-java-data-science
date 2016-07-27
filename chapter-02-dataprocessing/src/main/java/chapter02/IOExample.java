package chapter02;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class IOExample {

    public static void main(String[] args) throws IOException {
        rawJavaIOExample();
        bufferedReaderNIOExample();
        nioFilesExample();
    }

    public static void rawJavaIOExample() throws IOException {
        List<String> lines = new ArrayList<>();

        try (InputStream is = new FileInputStream("data/text.txt")) {
            try (InputStreamReader isReader = new InputStreamReader(is, StandardCharsets.UTF_8)) {
                try (BufferedReader reader = new BufferedReader(isReader)) {

                    while (true) {
                        String line = reader.readLine();
                        if (line == null) {
                            break;
                        }
                        lines.add(line);
                    }

                    isReader.close();
                }
            }
        }

        System.out.println(lines);
    }

    public static void bufferedReaderNIOExample() throws IOException {
        List<String> lines = new ArrayList<>();

        Path path = Paths.get("data/text.txt");
        try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            while (true) {
                String line = reader.readLine();
                if (line == null) {
                    break;
                }
                lines.add(line);
            }
        }

        System.out.println(lines);
    }

    public static void nioFilesExample() throws IOException {
        Path path = Paths.get("data/text.txt");
        List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
        System.out.println(lines);
    }

}
