package chapter02;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

public class CommonsCsvExample {

    public static void main(String[] args) throws IOException {
        List<Person> csv = csvExample();
        System.out.println(csv);

        List<Person> tsv = tsvExample();
        System.out.println(tsv);
    }

    public static List<Person> csvExample() throws IOException {
        List<Person> result = new ArrayList<>();

        Path csvFile = Paths.get("data/csv-example-generatedata_com.csv");
        try (BufferedReader reader = Files.newBufferedReader(csvFile, StandardCharsets.UTF_8)) {
            CSVFormat csv = CSVFormat.RFC4180.withHeader();
            try (CSVParser parser = csv.parse(reader)) {
                Iterator<CSVRecord> it = parser.iterator();
                // List<CSVRecord> records = parse.getRecords();
                it.forEachRemaining(rec -> {
                    String name = rec.get("name");
                    String email = rec.get("email");
                    String country = rec.get("country");
                    int salary = Integer.parseInt(rec.get("salary").substring(1));
                    int experience = Integer.parseInt(rec.get("experience"));
                    Person person = new Person(name, email, country, salary, experience);
                    result.add(person);
                });
            }
        }

        return result;
    }

    public static List<Person> tsvExample() throws IOException {
        List<Person> result = new ArrayList<>();

        Path csvFile = Paths.get("data/tsv-example-generatedata_com.tsv");
        try (BufferedReader reader = Files.newBufferedReader(csvFile, StandardCharsets.UTF_8)) {
            CSVFormat tsv = CSVFormat.TDF.withHeader();
            try (CSVParser parser = tsv.parse(reader)) {
                Iterator<CSVRecord> it = parser.iterator();
                it.forEachRemaining(rec -> {
                    String name = rec.get("name");
                    String email = rec.get("email");
                    String country = rec.get("country");
                    int salary = Integer.parseInt(rec.get("salary").substring(1));
                    int experience = Integer.parseInt(rec.get("experience"));
                    Person person = new Person(name, email, country, salary, experience);
                    result.add(person);
                });
            }
        }

        return result;
    }
}
