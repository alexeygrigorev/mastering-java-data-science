package chapter06.project;

import static org.junit.Assert.assertTrue;

import java.util.Arrays;

import org.junit.Test;

public class FirstPageOrNotTest {

    @Test
    public void concat1d() {
        double[] y1 = { 1.0, 2.0, 3.0 };
        double[] y2 = { 4.0, 5.0 };
        double[] y = FirstPageOrNot.concat(y1, y2);
        double[] expected = { 1.0, 2.0, 3.0, 4.0, 5.0 };
        assertTrue(Arrays.equals(y, expected));
    }

    @Test
    public void concat2d() {
        double[][] X1 = { { 0.0, 1.0 }, { 0.0, 2.0 }, { 0.0, 3.0 } };
        double[][] X2 = { { 0.0, 4.0 }, { 0.0, 5.0 } };
        double[][] X = FirstPageOrNot.concat(X1, X2);
        double[][] expected = { { 0.0, 1.0 }, { 0.0, 2.0 }, { 0.0, 3.0 }, { 0.0, 4.0 }, { 0.0, 5.0 } };
        assertTrue(Arrays.deepEquals(X, expected));
    }

}
