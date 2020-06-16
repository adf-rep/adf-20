package utils;

import java.util.Arrays;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

public class Commons {

    public static double[] concat(double[] a, double[] b) {
        return DoubleStream.concat(Arrays.stream(a), Arrays.stream(b)).toArray();
    }

}
