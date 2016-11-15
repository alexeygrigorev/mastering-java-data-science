package chapter06.ml;

import java.io.PrintStream;
import java.util.Iterator;

import org.apache.commons.io.output.NullOutputStream;

import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import smile.data.Datum;
import smile.data.SparseDataset;
import smile.math.SparseArray;
import smile.math.SparseArray.Entry;

public class SparseLibLinear {

    public static void mute() {
        PrintStream devNull = new PrintStream(new NullOutputStream());
        Linear.setDebugOutput(devNull);
    }

    public static Model train(SparseDataset dataset, double[] y, Parameter param) {
        Problem problem = wrapSparseDataset(dataset, y);
        return Linear.train(problem, param);
    }

    private static Problem wrapSparseDataset(SparseDataset dataset, double[] y) {
        Feature[][] X = wrapX(dataset);

        Problem problem = new Problem();
        problem.x = X;
        problem.y = y;
        problem.n = dataset.ncols() + 1;
        problem.l = dataset.size();

        return problem;
    }

    private static Feature[][] wrapX(SparseDataset dataset) {
        int nrow = dataset.size();
        Feature[][] X = new Feature[nrow][];
        int i = 0;

        for (Datum<SparseArray> inrow : dataset) {
            X[i] = wrapRow(inrow);
            i++;
        }

        return X;
    }

    private static Feature[] wrapRow(Datum<SparseArray> inrow) {
        SparseArray features = inrow.x;
        int nonzero = features.size();
        Feature[] outrow = new Feature[nonzero];

        Iterator<Entry> it = features.iterator();
        for (int j = 0; j < nonzero; j++) {
            Entry next = it.next();
            outrow[j] = new FeatureNode(next.i + 1, next.x);
        }

        return outrow;
    }

    public static double[] predictProba(Model model, SparseDataset dataset) {
        int n = dataset.size();

        double[] results = new double[n];
        double[] probs = new double[2];

        int i = 0;

        for (Datum<SparseArray> inrow : dataset) {
            Feature[] row = wrapRow(inrow);
            Linear.predictProbability(model, row, probs);
            results[i] = probs[1];
            i++;
        }

        return results;
    }

    public static double[] sigmoid(double[] scores) {
        double[] result = new double[scores.length];

        for (int i = 0; i < result.length; i++) {
            result[i] = 1 / (1 + Math.exp(-scores[i]));
        }

        return result;
    }
}
