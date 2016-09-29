package chapter05.dimred;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.DiagonalMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import com.google.common.base.Stopwatch;

import chapter05.cv.Dataset;
import chapter05.cv.Split;
import chapter05.preprocess.StandardizationPreprocessor;
import smile.math.matrix.Matrix;
import smile.math.matrix.SingularValueDecomposition;
import smile.regression.OLS;
import smile.regression.Regression;
import smile.validation.MSE;

public class PerformanceTruncatedSvd {

    public static void main(String[] args) throws IOException {
        Dataset dataset = PerformanceData.readData();

        StandardizationPreprocessor preprocessor = StandardizationPreprocessor.train(dataset);
        dataset = preprocessor.transform(dataset);

        Stopwatch stopwatch = Stopwatch.createStarted();
        Matrix matrix = new Matrix(dataset.getX());
        SingularValueDecomposition svd = SingularValueDecomposition.decompose(matrix, 100);
        double[][] projected = XV(dataset.getX(), svd.getV()); 
        System.out.println("SVD took " + stopwatch.stop());

        double[] explainedVariance = explainedVariance(dataset.getX(), svd);
        System.out.println(Arrays.toString(explainedVariance));

        dataset = new Dataset(projected, dataset.getY());

        Split trainTestSplit = dataset.shuffleSplit(0.3);
        Dataset train = trainTestSplit.getTrain();

        List<Split> folds = train.shuffleKFold(3);
        DescriptiveStatistics ols = crossValidate(folds, data -> {
            return new OLS(data.getX(), data.getY());
        });

        System.out.printf("ols: rmse=%.4f ± %.4f%n", ols.getMean(), ols.getStandardDeviation());
    }

    private static double[] explainedVariance(double[][] X, SingularValueDecomposition svd) {
        double totalVariance = totalVariance(X);
        int nrows = X.length;

        double[] singularValues = svd.getSingularValues();
        double[] cumulatedRatio = new double[singularValues.length];

        double acc = 0.0;
        for (int i = 0; i < singularValues.length; i++) {
            double s = singularValues[i];
            double ratio = (s * s / nrows) / totalVariance;
            acc = acc + ratio;
            cumulatedRatio[i] = acc;
        }

        return cumulatedRatio;
    }

    public static double totalVariance(double[][] X) {
        Array2DRowRealMatrix matrix = new Array2DRowRealMatrix(X, false);
        int ncols = matrix.getColumnDimension();

        double totalVariance = 0.0;
        for (int col = 0; col < ncols; col++) {
            double[] column = matrix.getColumn(col);
            DescriptiveStatistics stats = new DescriptiveStatistics(column);
            totalVariance = totalVariance + stats.getVariance();
        }

        return totalVariance;
    }

    public static double[][] US(SingularValueDecomposition svd) {
        DiagonalMatrix S = new DiagonalMatrix(svd.getSingularValues());
        Array2DRowRealMatrix U = new Array2DRowRealMatrix(svd.getU(), false);

        RealMatrix result = S.multiply(U.transpose()).transpose();
        return result.getData();
    }

    public static double[][] XV(double[][] Xd, double[][] Vd) {
        Array2DRowRealMatrix X = new Array2DRowRealMatrix(Xd, false);
        Array2DRowRealMatrix V = new Array2DRowRealMatrix(Vd, false);
        return X.multiply(V).getData();
    }

    public static DescriptiveStatistics crossValidate(List<Split> folds,
            Function<Dataset, Regression<double[]>> trainer) {
        double[] aucs = folds.parallelStream().mapToDouble(fold -> {
            Dataset train = fold.getTrain();
            Dataset validation = fold.getTest();
            Regression<double[]> model = trainer.apply(train);
            return rmse(model, validation);
        }).toArray();

        return new DescriptiveStatistics(aucs);
    }

    private static double rmse(Regression<double[]> model, Dataset dataset) {
        double[] prediction = predict(model, dataset);
        double[] truth = dataset.getY();

        double mse = new MSE().measure(truth, prediction);
        return Math.sqrt(mse);
    }

    public static double[] predict(Regression<double[]> model, Dataset dataset) {
        double[][] X = dataset.getX();
        double[] result = new double[X.length];

        for (int i = 0; i < X.length; i++) {
            result[i] = model.predict(X[i]);
        }

        return result;
    }
}
