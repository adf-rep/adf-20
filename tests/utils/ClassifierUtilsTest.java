package utils;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ClassifierUtilsTest {

    @Test
    void combinePredictionsMax() {
        assertEquals(0.55, ClassifierUtils.combinePredictionsMax(new double[]{0.0, 0.45, 0.55}));
        assertEquals(1.0, ClassifierUtils.combinePredictionsMax(new double[]{0.0, 1.0}));
        assertEquals(0.0, ClassifierUtils.combinePredictionsMax(new double[]{1.0}));
        assertEquals(0.6666, ClassifierUtils.combinePredictionsMax(new double[]{0.0, 0.5, 1.0}), 0.0001);
        assertEquals(0.5, ClassifierUtils.combinePredictionsMax(new double[]{10.0, 15.0, 25.0}));
    }

    @Test
    void prediction() {
        assertArrayEquals(
                new double[]{((0.1 * 0.9 + 0.2 * 0.2) / (0.9 + 0.2)), (0.5 * 0.9 + 0.4 * 0.2) / 1.1, (0.4 * 0.9 + 0.4 * 0.2) / 1.1},
                ClassifierUtils.prediction(new double[]{0.1, 0.5, 0.4, 0.2, 0.4, 0.4}, 3, new double[]{0.9, 0.2}),
                0.001);

        assertArrayEquals(
                new double[]{((0.1 + 0.2 ) / 2), (0.5 + 0.4) / 2, (0.4 + 0.4) / 2},
                ClassifierUtils.prediction(new double[]{0.1, 0.5, 0.4, 0.2, 0.4, 0.4}, 3),
                0.001);

        assertArrayEquals(
                new double[]{0.1, 0.5, 0.4},
                ClassifierUtils.prediction(new double[]{0.1, 0.5, 0.4}, 3),
                0.001);
    }

    @Test
    void complementPrediction() {
        assertArrayEquals(
                new double[]{0.2, 0.5, 0.05, 0.0, 0.0},
                ClassifierUtils.complementPrediction(new double[]{0.2, 0.5, 0.05}, 5),
                0.001);
    }

}