package chapter04.classification;

import java.awt.Color;
import java.awt.Dimension;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JFrame;
import javax.swing.WindowConstants;

import com.google.common.collect.Ordering;

import smile.plot.Line.Style;
import smile.plot.PlotCanvas;

public class RocCurve {

    public static void plot(double[] actual, double[] prediction) {
        List<ActualPredictedPair> pairs = sortByScore(actual, prediction);
        int pos = numberOfPositives(pairs);
        double[][] plotData = rocData(pairs, pos);
        rocPlot(plotData);
    }

    private static List<ActualPredictedPair> sortByScore(double[] actual, double[] prediction) {
        int length = actual.length;

        List<ActualPredictedPair> pairs = new ArrayList<>(length);
        for (int i = 0; i < length; i++) {
            pairs.add(new ActualPredictedPair(actual[i], prediction[i]));
        }

        Ordering<ActualPredictedPair> ordering =
                Ordering.natural().onResultOf(ActualPredictedPair::getPredicted);
        return ordering.immutableSortedCopy(pairs);
    }

    private static int numberOfPositives(List<ActualPredictedPair> pairs) {
        int pos = 0;

        for (ActualPredictedPair pair : pairs) {
            if (pair.isPositive()) {
                pos++;
            } else if (!pair.isNegative()) {
                throw new IllegalArgumentException("if a pair is not positive, it must be negative - but it's not");
            }
        }

        return pos;
    }

    private static double[][] rocData(List<ActualPredictedPair> pairs, int pos) {
        int n = pairs.size();

        int neg = n - pos;
        double stepUp = 1.0 / pos;
        double stepRight = 1.0 / neg;

        double tpr = 0.0;
        double fpr = 0.0;

        double[][] data = new double[n][2];

        for (int i = 0; i < n; i++) {
            ActualPredictedPair pair = pairs.get(i);
            if (pair.isPositive()) {
                tpr = tpr + stepUp;
            } else if (pair.isNegative()) {
                fpr = fpr + stepRight;
            }

            data[i][0] = fpr;
            data[i][1] = tpr;
        }

        return data;
    }

    private static void rocPlot(double[][] rocData) {
        JFrame frame = new JFrame("ROC Curve");
        frame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);

        double[] lowerBound = { 0, 0 };
        double[] upperBound = { 1, 1 };

        PlotCanvas canvas = new PlotCanvas(lowerBound, upperBound, false);
        canvas.setAxisLabel(0, "False Positive Rate");
        canvas.setAxisLabel(1, "True Positive Rate");
        canvas.setTitle("ROC Curve");

        double[][] baseline = { { 0, 0 }, { 1, 1 } };
        canvas.line(baseline, Style.DASH, Color.GRAY);
        canvas.line(rocData, Style.SOLID, Color.BLACK);

        frame.add(canvas);

        frame.setSize(new Dimension(1000, 1000));
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }

    private static class ActualPredictedPair {
        private final double actual;
        private final double predicted;

        public ActualPredictedPair(double actual, double predicted) {
            this.actual = actual;
            this.predicted = predicted;
        }

        public boolean isPositive() {
            return actual == 1.0;
        }

        public boolean isNegative() {
            return actual == 0.0;
        }

        public double getPredicted() {
            return predicted;
        }

        @Override
        public String toString() {
            return String.format("%4.3f (%1.0f)", predicted, actual);
        }
    }

}
