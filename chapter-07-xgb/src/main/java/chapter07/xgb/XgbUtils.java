package chapter07.xgb;

import java.util.ArrayList;
import java.util.List;

import ml.dmlc.xgboost4j.LabeledPoint;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;
import chapter07.cv.Dataset;

public class XgbUtils {

    public static double[] unwrapToDouble(float[][] floatResults) {
        int n = floatResults.length;
        double[] result = new double[n];
        for (int i = 0; i < n; i++) {
            result[i] = floatResults[i][0];
        }
        return result;
    }

    public static DMatrix wrapData(Dataset data) throws XGBoostError {
        int nrow = data.length();
        double[][] X = data.getX();
        double[] y = data.getY();
        List<LabeledPoint> points = new ArrayList<>();

        for (int i = 0; i < nrow; i++) {
            float label = (float) y[i];
            float[] floatRow = asFloat(X[i]);
            LabeledPoint point = LabeledPoint.fromDenseVector(label, floatRow);
            points.add(point);
        }

        String cacheInfo = "";
        return new DMatrix(points.iterator(), cacheInfo);
    }

    public static float[] asFloat(double[] ds) {
        float[] result = new float[ds.length];
        for (int i = 0; i < ds.length; i++) {
            result[i] = (float) ds[i];
        }
        return result;
    }

    public static double[] preduct(Booster model, DMatrix data) throws XGBoostError {
        float[][] predict = model.predict(data);
        return unwrapToDouble(predict);
    }

    public static double[] preduct(Booster model, Dataset data) throws XGBoostError {
        DMatrix dmatrix = wrapData(data);
        return preduct(model, dmatrix);
    }
}
