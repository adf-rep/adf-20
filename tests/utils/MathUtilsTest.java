package utils;

import com.yahoo.labs.samoa.instances.Instance;
import eval.experiment.ExperimentStream;
import moa.streams.ArffFileStream;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import utils.math.MathUtils;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class MathUtilsTest {

    private static List<Instance> test2DNumericInstances = new ArrayList<>();

    @BeforeAll
    static void loadStreams() {
        ExperimentStream test2DNumericStream = new ExperimentStream(new ArffFileStream("tests/data/test2Dnumeric.arff", 3), "TEST2D", 16, 1);
        while (test2DNumericStream.stream.hasMoreInstances()) {
            test2DNumericInstances.add(test2DNumericStream.stream.nextInstance().getData());
        }
    }

    @Test
    void euclideanDist() {
        Instance instanceA = test2DNumericInstances.get(0);
        Instance instanceB = test2DNumericInstances.get(1);
        assertEquals(1.0, MathUtils.euclideanDist(instanceA, instanceB));

        instanceA = test2DNumericInstances.get(4);
        instanceB = test2DNumericInstances.get(7);
        assertEquals(2.8284, MathUtils.euclideanDist(instanceA, instanceB), 0.0001);
    }

    @Test
    void randomExpNormal() {
        double lambda = 1;
        for (int i = 0; i < 100; i++) {
            double x = MathUtils.randomExpNormal(lambda);
            assertTrue(x >= 0.0 && x <= 1.0);
        }
    }
}