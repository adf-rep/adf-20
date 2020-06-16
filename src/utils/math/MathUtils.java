package utils.math;
import com.yahoo.labs.samoa.instances.Instance;
import javafx.util.Pair;

import java.util.ArrayList;
import java.util.stream.DoubleStream;

public class MathUtils {

    public static double euclideanDist(Instance a, Instance b) {
        double euclideanDist = 0.0;
        for (int i = 0; i < a.numAttributes() - 1; i++) {
            euclideanDist += Math.pow(a.value(i) - b.value(i), 2.0);
        }

        return Math.sqrt(euclideanDist);
    }

    public static int randomPoisson01(double lambda) {
        double L = Math.exp(-lambda);
        double t = 1.0;
        int k = 0;

        do {
            k++;
            t *= Math.random();
        } while (t > L);

        return (k > 1 ? k - 1 : k);
    }

    public static int randomZTP(double lambda) {
        int k = 1;
        double t = (Math.exp(-lambda) / (1 - Math.exp(-lambda))) * lambda;
        double s = t;
        double u = Math.random();

        while (s < u) {
            k += 1;
            t *= (lambda / k);
            s += t;
        }

        return k;
    }

    public static double randomExpNormal(double lambda) {
        double x = Math.random();
        double y = -Math.log(x) / lambda;

        return y % 1;
    }

    public static boolean randomBernoulli(double p) {
        return Math.random() <= p;
    }

    public static double[] multiplyVectors(double[] v1, double[] v2) {
        assert v1.length == v2.length;
        double[] result = new double[v1.length];

        for (int i = 0; i < result.length; i++) {
            result[i] = v1[i] * v2[i];
        }

        return result;
    }

    public static double sigmoid(double x, double beta) {
        return x*(beta - 1) / (2.0*beta*x - beta - 1);
    }

    public static double gmean(double sensitivity, double specificity) {
        return Math.sqrt(sensitivity * specificity);
    }

    public static double root(double num, double root) {
        return Math.pow(Math.E, Math.log(num)/root);
    }

    public static double max(double[] a) {
        double max = Double.MIN_VALUE;
        for (double aVal : a) {
            if (aVal > max) max = aVal;
        }

        return max;
    }

    public static Pair<Integer, Double> maxPair(double[] a) {
        double max = Double.MIN_VALUE;
        int maxIndex = -1;

        for (int i = 0; i < a.length; i++) {
            if (a[i] > max) {
                max = a[i];
                maxIndex = i;
            }
        }

        return new Pair<>(maxIndex, max);
    }

    public static double sum(ArrayList<Double> v) {
        return v.stream().reduce(0.0, Double::sum);
    }

    public static double sum(double[] v) {
        return DoubleStream.of(v).sum();
    }

    public static double incMean(double mean, double v, int n) {
        return mean + (v - mean) / n;
    }

    public static double mergeMean(double mean, int count, double otherMean, int otherCount) {
        return (count * mean + otherCount * otherMean) / (count + otherCount);
    }

    public static double incVar(double var, double prevMean, double v, int n) {
        if (n == 1) return 0.0;
        return ((n - 2.0) / (n - 1.0)) * var + (1.0 / n) * Math.pow(v - prevMean, 2.0);
    }

    public static double mergeVarPool(double var, int count, double otherVar, int otherCount) {
        return ((count - 1.0) * var + (otherCount - 1) * otherVar) / (count + otherCount - 2.0);
    }

    public static double log(double x, double base) {
        return Math.log(x) / Math.log(base);
    }

}
