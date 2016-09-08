package chapter04;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import com.fasterxml.jackson.jr.ob.JSON;
import com.google.common.base.Throwables;

import chapter04.cv.Dataset;
import chapter04.cv.Fold;
import joinery.DataFrame;

public class RankedPageData {

    public static List<RankedPage> readRankedPages() throws IOException {
        Path path = Paths.get("./data/ranked-pages.json");
        try (Stream<String> lines = Files.lines(path)) {
            return lines.map(line -> parseJson(line)).collect(Collectors.toList());
        }
    }

    public static RankedPage parseJson(String line) {
        try {
            return JSON.std.beanFrom(RankedPage.class, line);
        } catch (IOException e) {
            throw Throwables.propagate(e);
        }
    }

    public static Fold readRankedPagesMatrix() throws IOException {
        List<RankedPage> pages = RankedPageData.readRankedPages();
        DataFrame<Object> dataframe = BeanToJoinery.convert(pages, RankedPage.class);

        List<Object> page = dataframe.col("page");
        double[] target = page.stream().mapToInt(o -> (int) o).mapToDouble(p -> (p == 0) ? 1.0 : 0.0).toArray();

        dataframe = dataframe.drop("page", "url", "position");
        double[][] X = dataframe.toModelMatrix(0.0);

        Dataset dataset = new Dataset(X, target);
        return dataset.trainTestSplit(0.2);
    }

}
