package cls.df;

import com.yahoo.labs.samoa.instances.Instance;
import eval.experiment.ExperimentStream;
import moa.streams.ArffFileStream;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class InputLayerTest {

    private static List<Instance> testStreamInstances = new ArrayList<>();

    @BeforeAll
    static void loadStreams() {
        ExperimentStream testStream = new ExperimentStream(new ArffFileStream("tests/data/testFull.arff", 19), "TEST", 10, 1);
        while (testStream.stream.hasMoreInstances()) {
            testStreamInstances.add(testStream.stream.nextInstance().getData());
        }
    }

    @Test
    void transform() {
        Instance instance = testStreamInstances.get(0);
        int size = instance.numAttributes() - 1;

        InputLayer inputLayer = new InputLayer(new LayerSize(2,10, 3), 1, false, true, false);
        ArrayList<double[]> grainRepresentations = inputLayer.transform(instance, true);
        assertEquals(3, grainRepresentations.size());

        int grainInstancesNum = size - (size / (int)Math.pow(2, 3)) + 1;
        assertEquals(grainInstancesNum * 5 * 2, grainRepresentations.get(0).length); // 17 * 10
        grainInstancesNum = size - (size / (int)Math.pow(2, 2)) + 1;
        assertEquals(grainInstancesNum * 5 * 2, grainRepresentations.get(1).length); // 15 * 10
        grainInstancesNum = size - (size / (int)Math.pow(2, 1)) + 1;
        assertEquals(grainInstancesNum * 5 * 2, grainRepresentations.get(2).length); // 10 * 10

        inputLayer = new InputLayer(new LayerSize(2,10, 3), -1, false, true, false);
        grainRepresentations = inputLayer.transform(instance, true);
        assertEquals(3, grainRepresentations.size());

        grainInstancesNum = 9;
        assertEquals(grainInstancesNum * 5 * 2, grainRepresentations.get(0).length); // 9 * 10
        grainInstancesNum = 5;
        assertEquals(grainInstancesNum * 5 * 2, grainRepresentations.get(1).length); // 5 * 10
        grainInstancesNum = 2;
        assertEquals(grainInstancesNum * 5 * 2, grainRepresentations.get(2).length); // 2 * 10
    }

}
