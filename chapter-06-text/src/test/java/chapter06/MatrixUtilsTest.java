package chapter06;

import static org.junit.Assert.*;

import java.util.Arrays;

import org.junit.Test;

public class MatrixUtilsTest {

    @Test
    public void slicedSimilarityEven() {
        double[][] matrix = { {1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}  };
        double[] vector = {0, 1};
        int sliceSize = 2;
        double[] similarity = MatrixUtils.slicedSimilarity(matrix, vector, sliceSize);

        double[] expected = {1, 2, 3, 4, 5, 6};

        assertTrue(Arrays.equals(expected, similarity));
    }

    @Test
    public void slicedSimilarityUneven() {
        double[][] matrix = { {1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {1, 7},  };
        double[] vector = {0, 1};
        int sliceSize = 3;
        double[] similarity = MatrixUtils.slicedSimilarity(matrix, vector, sliceSize);

        double[] expected = {1, 2, 3, 4, 5, 6, 7};

        assertTrue(Arrays.equals(expected, similarity));
    }

}
