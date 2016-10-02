package chapter05.dimred;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.apache.commons.lang3.SerializationUtils;

import com.google.common.base.Stopwatch;

import joinery.DataFrame;

public class Categorical {
    public static DataFrame<Object> readData() throws IOException {
        Stopwatch stopwatch = Stopwatch.createStarted();

        Path path = Paths.get("data/categorical.bin");

        if (path.toFile().exists()) {
            try (InputStream os = Files.newInputStream(path)) {
                DfHolder holder = SerializationUtils.deserialize(os);
                DataFrame<Object> df = holder.toDf();
                System.out.println("reading dataframe from cache took " + stopwatch.stop());
                return df;
            }
        }

        DataFrame<Object> dataframe = DataFrame.readCsv("data/consumer_complaints.csv");
        System.out.println("reading dataframe took " + stopwatch.stop());

        DataFrame<Object> categorical = dataframe.retain("product", "sub_product", "issue", 
                "sub_issue", "company_public_response", "company", 
                "state", "zipcode", "consumer_consent_provided", 
                "submitted_via");

        try (OutputStream os = Files.newOutputStream(path)) {
            DfHolder holder = new DfHolder(categorical);
            SerializationUtils.serialize(holder, os);
        }

        return categorical;
    }
}
