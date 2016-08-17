package chapter03;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;
import java.util.stream.Collectors;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

import com.google.common.collect.Maps;

public class CommonsMathExample {

    public static void main(String[] args) throws IOException {
        List<RankedPage> data = Data.readRankedPages();
        summaryStats(data);
        descStats(data);
        proportion(data);
        groupByDescStats(data);
    }

    private static void summaryStats(List<RankedPage> data) {
        SummaryStatistics summary = new SummaryStatistics();
        data.stream().mapToDouble(RankedPage::getBodyContentLength).forEach(summary::addValue);
        System.out.println(summary.getSummary());
    }

    private static void descStats(List<RankedPage> data) {
        double[] dataArray = data.stream().mapToDouble(RankedPage::getBodyContentLength).toArray();
        DescriptiveStatistics desc = new DescriptiveStatistics(dataArray);
        System.out.printf("min: %9.1f%n", desc.getMin());
        System.out.printf("p05: %9.1f%n", desc.getPercentile(5));
        System.out.printf("p25: %9.1f%n", desc.getPercentile(25));
        System.out.printf("p50: %9.1f%n", desc.getPercentile(50));
        System.out.printf("p75: %9.1f%n", desc.getPercentile(75));
        System.out.printf("p95: %9.1f%n", desc.getPercentile(95));
        System.out.printf("max: %9.1f%n", desc.getMax());
    }

    private static void proportion(List<RankedPage> data) {
        double proportion = data.stream()
                .mapToInt(p -> p.getBodyContentLength() == 0 ? 1 : 0)
                .average().getAsDouble();
        System.out.printf("proportion of zero content length: %.5f%n", proportion);
    }

    private static void groupByDescStats(List<RankedPage> data) {
        System.out.println();

        Map<Integer, List<RankedPage>> byPage = data.stream()
                .filter(p -> p.getBodyContentLength() != 0)
                .collect(Collectors.groupingBy(RankedPage::getPage));

        List<DescriptiveStatistics> stats = byPage.entrySet().stream()
                .sorted(Map.Entry.comparingByKey())
                .map(e -> calculate(e.getValue(), RankedPage::getBodyContentLength))
                .collect(Collectors.toList());

        Map<String, Function<DescriptiveStatistics, Double>> functions = Maps.newLinkedHashMap();
        functions.put("min", d -> d.getMin());
        functions.put("p05", d -> d.getPercentile(5));
        functions.put("p25", d -> d.getPercentile(25));
        functions.put("p50", d -> d.getPercentile(50));
        functions.put("p75", d -> d.getPercentile(75));
        functions.put("p95", d -> d.getPercentile(95));
        functions.put("max", d -> d.getMax());

        System.out.print("page");
        for (Integer page : byPage.keySet()) {
            System.out.printf("%9d ", page);
        }
        System.out.println();

        for (Entry<String, Function<DescriptiveStatistics, Double>> pair : functions.entrySet()) {
            System.out.print(pair.getKey());
            Function<DescriptiveStatistics, Double> function = pair.getValue();
            System.out.print(" ");
            for (DescriptiveStatistics ds : stats) {
                System.out.printf("%9.1f ", function.apply(ds));
            }
            System.out.println();
        }
    }

    private static DescriptiveStatistics calculate(List<RankedPage> data, ToDoubleFunction<RankedPage> getter) {
        double[] dataArray = data.stream().mapToDouble(getter).toArray();
        return new DescriptiveStatistics(dataArray);
    }

}
