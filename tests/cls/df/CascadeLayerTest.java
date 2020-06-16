package cls.df;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import eval.experiment.ExperimentStream;
import moa.streams.ArffFileStream;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class CascadeLayerTest {

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
        InputLayer inputLayer = new InputLayer(new LayerSize(2,10, 3), 1, false, true, false);
        ArrayList<double[]> grainRepresentations = inputLayer.transform(instance, true);

        CascadeLayer cascadeLayer = new CascadeLayer(new LayerSize(2, 10, 3), false, true, true, true);
        double[] layerOutput = cascadeLayer.transform(new double[0], grainRepresentations, instance, true);
        assertEquals(2 * 5, layerOutput.length);

        cascadeLayer = new CascadeLayer(new LayerSize(10, 10, 3), false, true, true, true);
        layerOutput = cascadeLayer.transform(new double[0], grainRepresentations, instance, true);
        assertEquals(10 * 5, layerOutput.length);

        cascadeLayer = new CascadeLayer(new LayerSize(2, 10, 3), true, true, true, true);
        layerOutput = cascadeLayer.transform(new double[0], grainRepresentations, instance, true);
        assertEquals(2 * 5, layerOutput.length);
    }

    @Test
    void extendInput() {
        Instance instance = testStreamInstances.get(5);
        InputLayer inputLayer = new InputLayer(new LayerSize(2,10, 3), 1, false, true, false);
        inputLayer.transform(instance, true);
        ArrayList<double[]> grainRepresentations = inputLayer.transform(instance, true);
        CascadeLayer cascadeLayer = new CascadeLayer(new LayerSize(2, 10, 3), false, true, true, true);
        cascadeLayer.transform(new double[0], grainRepresentations, instance, true);
        double[] layerOutput = cascadeLayer.transform(new double[0], grainRepresentations, instance, true);

        Instance subLayerInput = CascadeLayer.extendInput(layerOutput, grainRepresentations.get(0), instance, true);
        assertEquals(layerOutput.length + grainRepresentations.get(0).length + instance.numAttributes(), subLayerInput.numAttributes());
        assertEquals((2 * 5) + (17 * 5 * 2 ) + 18 + 1, subLayerInput.numAttributes()); // 199
        assertEquals(instance.classValue(), subLayerInput.classValue());
        System.out.println(Arrays.toString(subLayerInput.toDoubleArray()));

        assertArrayEquals(new double[] {
                // layerInput: 2 * 5 = 10
                1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0,

                // grainRepresentation: 17 * 5 * 2 = 170
                1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0,
                1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0,
                1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0,
                1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0,
                1000.0, 0.0, 0.0, 0.0, 0.0,

                1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0,
                1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0,
                1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0,
                1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0,
                1000.0, 0.0, 0.0, 0.0, 0.0,

                // instance: 18 + 1 = 19
                0.7349, 0.1952, 1.0, 4.0, 7.0, 0.0, 0.5244, 0.1704, 0.9395, 0.1558, 0.2676, 0.4509, 0.4848, 0.1357, 0.5002, 0.31, 0.8081, 0.6956, 0.0
                }, subLayerInput.toDoubleArray());

        for (int i = 0; i < layerOutput.length; i++) {
            assertTrue(subLayerInput.attribute(i).toString().contains("prev_out"));
        }
        for (int i = 0; i < grainRepresentations.get(0).length; i++) {
            assertTrue(subLayerInput.attribute(i + layerOutput.length).toString().contains("grain_rep"));
        }
        for (int i = 0; i < instance.numAttributes(); i++) {
            assertEquals(subLayerInput.attribute(i + layerOutput.length + grainRepresentations.get(0).length).toString(), instance.attribute(i).toString());
        }

        subLayerInput = CascadeLayer.extendInput(new double[0], grainRepresentations.get(0), instance, true);
        assertEquals(grainRepresentations.get(0).length + instance.numAttributes(), subLayerInput.numAttributes());
        assertEquals(instance.classValue(), subLayerInput.classValue());

        subLayerInput = CascadeLayer.extendInput(layerOutput, grainRepresentations.get(0), instance, false);
        assertEquals(layerOutput.length + grainRepresentations.get(0).length + 1, subLayerInput.numAttributes());
        assertEquals((2 * 5) + (17 * 5 * 2 ) + 1, subLayerInput.numAttributes()); // 171
        assertEquals(instance.classValue(), subLayerInput.classValue());

        assertArrayEquals(new double[] {
                // layerInput: 2 * 5 = 10
                1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0,

                // grainRepresentation: 17 * 5 * 2 = 170
                1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0,
                1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0,
                1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0,
                1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0,
                1000.0, 0.0, 0.0, 0.0, 0.0,

                1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0,
                1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0,
                1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0,
                1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0,
                1000.0, 0.0, 0.0, 0.0, 0.0,

                // instance class: 1
                0.0
        }, subLayerInput.toDoubleArray());

        subLayerInput = CascadeLayer.extendInput(new double[0], grainRepresentations.get(0), instance, false);
        assertEquals(grainRepresentations.get(0).length + 1, subLayerInput.numAttributes());
        assertEquals(instance.classValue(), subLayerInput.classValue());

        subLayerInput = CascadeLayer.extendInput(layerOutput, new double[0], instance, true);
        assertEquals(layerOutput.length + instance.toDoubleArray().length, subLayerInput.numAttributes());
        assertEquals((2 * 5) + 19, subLayerInput.numAttributes()); // 29
        assertEquals(instance.classValue(), subLayerInput.classValue());

        assertArrayEquals(new double[] {
                // layerInput: 2 * 5 = 10
                1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0,

                // instance: 18 + 1 = 19
                0.7349, 0.1952, 1.0, 4.0, 7.0, 0.0, 0.5244, 0.1704, 0.9395, 0.1558, 0.2676, 0.4509, 0.4848, 0.1357, 0.5002, 0.31, 0.8081, 0.6956, 0.0
        }, subLayerInput.toDoubleArray());
    }

}
