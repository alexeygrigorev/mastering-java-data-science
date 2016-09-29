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
import smile.data.SparseDataset;
import smile.math.matrix.SingularValueDecomposition;
import smile.math.matrix.SparseMatrix;

public class CategoricalSvd {

    public static void main(String[] args) throws IOException {
        DataFrame<Object> categorical = readCategoricalData();

        Stopwatch stopwatch = Stopwatch.createStarted();
        SparseDataset sparse = OHE.hashingEncoding(categorical, 50_000);// oneHotEncoding(categorical);
        System.out.println("OHE took " + stopwatch.stop());

        stopwatch = Stopwatch.createStarted();
        SparseMatrix matrix = sparse.toSparseMatrix();
        SingularValueDecomposition svd = SingularValueDecomposition.decompose(matrix, 100);
        System.out.println("SVD took " + stopwatch.stop());

        System.out.println("V dim: " + svd.getV().length + " x " + svd.getV()[0].length);

        stopwatch = Stopwatch.createStarted();
        double[][] proj = Projections.project(sparse, svd.getV());
        System.out.println("projection: " + proj.length + " x " + proj[0].length + ", " + stopwatch.stop());
    }

    private static DataFrame<Object> readCategoricalData() throws IOException {
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
