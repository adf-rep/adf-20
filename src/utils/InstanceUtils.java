package utils;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import javafx.util.Pair;
import utils.math.MathUtils;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class InstanceUtils {

    public static Instances createInstances(Instance templateInstance, boolean insert) {
        Instances newInstances = new Instances(templateInstance.copy().dataset());

        if (insert) {
            newInstances.add(templateInstance.copy());
        }

        return newInstances;
    }

    public static List<Integer> instanceIndices(Instances nearestNeighbors) {
        return Arrays.stream(IntStream.range(0, nearestNeighbors.numInstances()).toArray())
                .boxed()
                .collect(Collectors.toList());
    }

    public static Pair<Integer, Double> findInstance(Instance instance, Instances instances, boolean closest) {
        double dist = (closest ? Double.MAX_VALUE : Double.MIN_VALUE);
        int idx = -1;

        for (int i = 0; i < instances.numInstances(); i++) {
            double otherDist = MathUtils.euclideanDist(instance, instances.get(i));
            if (closest ? otherDist < dist : otherDist > dist) {
                dist = otherDist;
                idx = i;
            }
        }

        return new Pair<>(idx, dist);
    }

    public static int[] getClosestIndices(Instance instance, Instances otherInstances, int indicesNum) {
        int n = otherInstances.numInstances();
        Double[] distances = new Double[n];

        for (int i = 0; i < n; i++) {
            distances[i] = MathUtils.euclideanDist(instance, otherInstances.get(i));
        }

        return IntStream.range(0, n).boxed()
                .sorted(Comparator.comparing(i -> distances[i]))
                .mapToInt(ele -> ele)
                .limit(indicesNum)
                .toArray();
    }


}
