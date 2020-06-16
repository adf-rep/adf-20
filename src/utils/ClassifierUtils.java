package utils;
import moa.core.DoubleVector;
import moa.core.Utils;

import java.util.Arrays;

public class ClassifierUtils {

    static public double combinePredictionsMax(double[] predictionValues) {
        double outPosterior;

        if (predictionValues.length > 1) {
            DoubleVector vote = new DoubleVector(predictionValues);

            if (vote.sumOfValues() > 0.0) {
                vote.normalize();
            }

            predictionValues = vote.getArrayRef();
            outPosterior = predictionValues[Utils.maxIndex(predictionValues)];

        } else {
            outPosterior = 0.0;
        }

        return Double.isInfinite(outPosterior) || Double.isNaN(outPosterior) ? 0 : outPosterior;
    }

    static public double[] prediction(double[] output, int numClasses, double[] weights) {
        double[] prediction = new double[numClasses];
        double n = 0;

        for (int i = 0; i < output.length; i += numClasses) {
            n += weights[i / numClasses];
            for (int j = 0; j < numClasses; j++) {
                prediction[j] = prediction[j] + (weights[i / numClasses] / n) * (output[i + j] - prediction[j]);
            }
        }

        return prediction;
    }

    static public double[] prediction(double[] output, int numClasses) {
        double[] weights = new double[output.length / numClasses];
        Arrays.fill(weights, 1.0);
        return prediction(output, numClasses, weights);
    }

    static public double[] complementPrediction(double[] incompletePrediction, int numClasses) {
        if (incompletePrediction.length == numClasses) return incompletePrediction;
        double[] completePrediction = new double[numClasses];
        System.arraycopy(incompletePrediction, 0, completePrediction, 0, incompletePrediction.length);
        return completePrediction;
    }

}
