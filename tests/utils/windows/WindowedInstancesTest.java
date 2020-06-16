package utils.windows;

import com.yahoo.labs.samoa.instances.Instance;
import eval.experiment.ExperimentStream;
import moa.streams.ArffFileStream;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class WindowedInstancesTest {

    private static List<Instance> test2DNumericInstances = new ArrayList<>();

    @BeforeAll
    static void loadStreams() {
        ExperimentStream test2DNumericStream = new ExperimentStream(new ArffFileStream("tests/data/test2Dnumeric.arff", 3), "TEST2D", 16, 1);
        while (test2DNumericStream.stream.hasMoreInstances()) {
            test2DNumericInstances.add(test2DNumericStream.stream.nextInstance().getData());
        }
    }

    @Test
    void add() {
        WindowedInstances win = new WindowedInstances(3, test2DNumericInstances.get(0), false);
        assertEquals(0, win.getWindowLength());

        win.add(test2DNumericInstances.get(0));
        assertEquals(1, win.getWindowLength());

        win.add(test2DNumericInstances.get(1));
        win.add(test2DNumericInstances.get(2));
        assertEquals(3, win.getWindowLength());

        win.add(test2DNumericInstances.get(3));
        assertEquals(3, win.getWindowLength());

        win = new WindowedInstances(3, test2DNumericInstances.get(0), true);
        assertEquals(1, win.getWindowLength());
    }

    @Test
    void getCentroid() {
        WindowedInstances win = new WindowedInstances(3, test2DNumericInstances.get(0), false).setKeepStatistics(true);

        win.add(test2DNumericInstances.get(4));
        Instance centroid = win.getCentroid();
        assertEquals(0.0, centroid.value(0));
        assertEquals(2.0, centroid.value(1));

        win.add(test2DNumericInstances.get(5));
        win.add(test2DNumericInstances.get(6));
        centroid = win.getCentroid();
        assertEquals(1.333, centroid.value(0), 0.001);
        assertEquals(1.666, centroid.value(1), 0.001);

        win.add(test2DNumericInstances.get(7));
        centroid = win.getCentroid();
        assertEquals(2.0, centroid.value(0), 0.1);
        assertEquals(1.0, centroid.value(1), 0.1);

        win = new WindowedInstances(3, test2DNumericInstances.get(0), true);
        centroid = win.getCentroid();
        assertEquals(0.0, centroid.value(0));
        assertEquals(1.0, centroid.value(1));
    }

    @Test
    void copy() {
        WindowedInstances win = new WindowedInstances(5, test2DNumericInstances.get(0), false);
        for (int i = 0; i < 4; i++) win.add(test2DNumericInstances.get(0));

        WindowedInstances winCopy = win.copy();
        assertEquals(win.getWindowLength(), winCopy.getWindowLength());

        winCopy.remove(0);
        assertEquals(3, winCopy.getWindowLength());
        assertEquals(4, win.getWindowLength());
    }

}