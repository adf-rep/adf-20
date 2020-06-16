package cls.df;

import cls.df.MultiGrainScanner;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import eval.experiment.ExperimentStream;
import moa.streams.ArffFileStream;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class MultiGrainScanTest {

    private static List<Instance> testStreamInstances = new ArrayList<>();
    private static List<Instance> testStreamImageInstances = new ArrayList<>();

    @BeforeAll
    static void loadStreams() {
        ExperimentStream testStream = new ExperimentStream(new ArffFileStream("tests/data/testFull.arff", 19), "TEST", 10, 1);
        while (testStream.stream.hasMoreInstances()) {
            testStreamInstances.add(testStream.stream.nextInstance().getData());
        }

        ExperimentStream testStreamImage = new ExperimentStream(new ArffFileStream("tests/data/testImage.arff", 26), "TEST", 20, 1);
        while (testStreamImage.stream.hasMoreInstances()) {
            testStreamImageInstances.add(testStreamImage.stream.nextInstance().getData());
        }
    }

    @Test
    void scan1D() {
        Instance instance = testStreamInstances.get(9);
        int size = instance.numAttributes() - 1;

        MultiGrainScanner mgs = new MultiGrainScanner(3, 1, false); // overlapping
        ArrayList<Instances> multiGrainInstances = mgs.scanInstance(instance);
        assertEquals(3, multiGrainInstances.size());

        assertEquals(size - (size / (int)Math.pow(2, 3)) + 1, multiGrainInstances.get(0).numInstances());
        for (int i = 0; i < multiGrainInstances.get(0).size(); i++) {
            assertEquals(size / (int)Math.pow(2, 3) + 1, multiGrainInstances.get(0).get(i).numAttributes());
            assertEquals(size / (int)Math.pow(2, 3), multiGrainInstances.get(0).get(i).classIndex());
            assertEquals(5, multiGrainInstances.get(0).get(i).numClasses());
            assertEquals(multiGrainInstances.get(0).get(i).classValue(), instance.classValue());
        }

        assertEquals(size - (size / (int)Math.pow(2, 2)) + 1, multiGrainInstances.get(1).numInstances());
        for (int i = 0; i < multiGrainInstances.get(1).size(); i++) {
            assertEquals(size / (int)Math.pow(2, 2) + 1, multiGrainInstances.get(1).get(i).numAttributes());
            assertEquals(size / (int)Math.pow(2, 2), multiGrainInstances.get(1).get(i).classIndex());
            assertEquals(5, multiGrainInstances.get(1).get(i).numClasses());
            assertEquals(multiGrainInstances.get(1).get(i).classValue(), instance.classValue());
        }

        assertEquals(size - (size / (int)Math.pow(2, 1)) + 1, multiGrainInstances.get(2).numInstances());
        for (int i = 0; i < multiGrainInstances.get(2).size(); i++) {
            assertEquals(size / (int)Math.pow(2, 1) + 1, multiGrainInstances.get(2).get(i).numAttributes());
            assertEquals(size / (int)Math.pow(2, 1), multiGrainInstances.get(2).get(i).classIndex());
            assertEquals(5, multiGrainInstances.get(2).get(i).numClasses());
            assertEquals(multiGrainInstances.get(2).get(i).classValue(), instance.classValue());
        }

        mgs = new MultiGrainScanner(3, -1, false); // non-overlapping
        multiGrainInstances = mgs.scanInstance(instance);
        assertEquals(3, multiGrainInstances.size());

        assertEquals(9, multiGrainInstances.get(0).numInstances());
        for (int i = 0; i < multiGrainInstances.get(0).size(); i++) {
            assertEquals(2 + 1, multiGrainInstances.get(0).get(i).numAttributes()); // 18 / 8 = 2
            for (int j = 0; j < 2; j++) assertEquals(i * 2 + j + 1, multiGrainInstances.get(0).get(i).value(j));

            assertEquals(2, multiGrainInstances.get(0).get(i).classIndex());
            assertEquals(multiGrainInstances.get(0).get(i).classValue(), instance.classValue());
        }

        assertEquals(5, multiGrainInstances.get(1).numInstances());
        for (int i = 0; i < multiGrainInstances.get(1).size(); i++) {
            assertEquals(4 + 1, multiGrainInstances.get(1).get(i).numAttributes()); // 18 / 4 = 4

            assertEquals(4, multiGrainInstances.get(1).get(i).classIndex());
            assertEquals(multiGrainInstances.get(1).get(i).classValue(), instance.classValue());
        }

        assertEquals(5, multiGrainInstances.get(1).get(1).value(0));
        assertEquals(6, multiGrainInstances.get(1).get(1).value(1));
        assertEquals(7, multiGrainInstances.get(1).get(1).value(2));
        assertEquals(8, multiGrainInstances.get(1).get(1).value(3));

        assertEquals(17, multiGrainInstances.get(1).get(4).value(0));
        assertEquals(18, multiGrainInstances.get(1).get(4).value(1));
        assertEquals(0, multiGrainInstances.get(1).get(4).value(2));
        assertEquals(0, multiGrainInstances.get(1).get(4).value(3));

        assertEquals(2, multiGrainInstances.get(2).numInstances());
        for (int i = 0; i < multiGrainInstances.get(2).size(); i++) {
            assertEquals(9 + 1, multiGrainInstances.get(2).get(i).numAttributes()); // 18 / 2 = 9
            for (int j = 0; j < 9; j++) assertEquals(i * 9 + j + 1, multiGrainInstances.get(2).get(i).value(j));

            assertEquals(9, multiGrainInstances.get(2).get(i).classIndex());
            assertEquals(multiGrainInstances.get(2).get(i).classValue(), instance.classValue());
        }
    }

    @Test
    void scan2D() {
        Instance instance = testStreamImageInstances.get(19);

        MultiGrainScanner mgs = new MultiGrainScanner(2, 1, true); // overlapping
        ArrayList<Instances> multiGrainInstances = mgs.scanInstance(instance);
        assertEquals(2, multiGrainInstances.size());

        assertEquals(5 * 5, multiGrainInstances.get(0).numInstances());
        for (int i = 0; i < multiGrainInstances.get(0).size(); i++) {
            assertEquals(1 * 1 + 1, multiGrainInstances.get(0).get(i).numAttributes()); // 5 / 4 = 1
            assertEquals(i + 1, multiGrainInstances.get(0).get(i).value(0));
            assertEquals(1, multiGrainInstances.get(0).get(i).classIndex());
            assertEquals(5, multiGrainInstances.get(0).get(i).numClasses());
            assertEquals(instance.classValue(), multiGrainInstances.get(0).get(i).classValue());
        }

        assertEquals(4 * 4, multiGrainInstances.get(1).numInstances());
        for (int i = 0; i < multiGrainInstances.get(1).size(); i++) {
            assertEquals(2 * 2 + 1, multiGrainInstances.get(1).get(i).numAttributes()); // 5 / 2 = 2
            //System.out.println(Arrays.toString(multiGrainInstances.get(1).get(i).toDoubleArray()));
            assertEquals((i % 4) + 1 + 5 * (i / 4), multiGrainInstances.get(1).get(i).value(0));
            assertEquals((i % 4) + 2 + 5 * (i / 4), multiGrainInstances.get(1).get(i).value(1));
            assertEquals((i % 4) + 1 + 5 * (1 + i / 4), multiGrainInstances.get(1).get(i).value(2));
            assertEquals((i % 4) + 2 + 5 * (1 + i / 4), multiGrainInstances.get(1).get(i).value(3));

            assertEquals(2 * 2, multiGrainInstances.get(1).get(i).classIndex());
            assertEquals(5, multiGrainInstances.get(1).get(i).numClasses());
            assertEquals(multiGrainInstances.get(1).get(i).classValue(), instance.classValue());
        }

        mgs = new MultiGrainScanner(2, -1, true); // non-overlapping
        multiGrainInstances = mgs.scanInstance(instance);
        assertEquals(2, multiGrainInstances.size());

        assertEquals(5 * 5, multiGrainInstances.get(0).numInstances());
        for (int i = 0; i < multiGrainInstances.get(0).size(); i++) {
            assertEquals(1 * 1 + 1, multiGrainInstances.get(0).get(i).numAttributes()); // 5 / 4 = 1
            assertEquals(i + 1, multiGrainInstances.get(0).get(i).value(0));
            assertEquals(1, multiGrainInstances.get(0).get(i).classIndex());
            assertEquals(5, multiGrainInstances.get(0).get(i).numClasses());
            assertEquals(instance.classValue(), multiGrainInstances.get(0).get(i).classValue());
        }

        assertEquals(3 * 3, multiGrainInstances.get(1).numInstances());
        for (int i = 0; i < multiGrainInstances.get(1).size(); i++) {
            assertEquals(2 * 2 + 1, multiGrainInstances.get(1).get(i).numAttributes()); // 5 / 2 = 2
            //System.out.println(Arrays.toString(multiGrainInstances.get(1).get(i).toDoubleArray()));
            assertEquals(2 * 2, multiGrainInstances.get(1).get(i).classIndex());
            assertEquals(5, multiGrainInstances.get(1).get(i).numClasses());
            assertEquals(multiGrainInstances.get(1).get(i).classValue(), instance.classValue());
        }

        assertEquals(3, multiGrainInstances.get(1).get(1).value(0));
        assertEquals(4, multiGrainInstances.get(1).get(1).value(1));
        assertEquals(8, multiGrainInstances.get(1).get(1).value(2));
        assertEquals(9, multiGrainInstances.get(1).get(1).value(3));

        assertEquals(5, multiGrainInstances.get(1).get(2).value(0));
        assertEquals(0, multiGrainInstances.get(1).get(2).value(1));
        assertEquals(10, multiGrainInstances.get(1).get(2).value(2));
        assertEquals(0, multiGrainInstances.get(1).get(2).value(3));

    }

}
