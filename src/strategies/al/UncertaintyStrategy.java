package strategies.al;
import com.yahoo.labs.samoa.instances.Instance;
import utils.ClassifierUtils;
import utils.Trackable;

import java.util.ArrayList;
import java.util.HashMap;

public class UncertaintyStrategy extends ActiveLearningStrategy implements Trackable {

    private double fixedThreshold = 0.3;
    private double variableThreshold = 1.0;
    private double variableThresholdStep = 0.01;
    private UncertaintyStrategyType strategy;

    private HashMap<String, Double> trackableParameters = new HashMap<>();

    public UncertaintyStrategy(UncertaintyStrategyType strategy, double budget) {
        this.strategy = strategy;
        this.budget = budget;
    }

    public UncertaintyStrategy(UncertaintyStrategyType strategy, double budget, double fixedThreshold,
                               double variableThresholdStep) {
        this.strategy = strategy;
        this.budget = budget;
        this.fixedThreshold = fixedThreshold;
        this.variableThresholdStep = variableThresholdStep;
    }

    @Override
    public void reset() {
        super.reset();
        this.variableThreshold = 1.0;
        this.trackableParameters.clear();
    }

    @Override
    public boolean queryLabel(Instance instance, double[] predictionValues, HashMap<String, Double> driftIndicators) {
        this.allInstances++;
        this.currentCost = this.labeledInstances / (double) ++this.iterationCount;
        if (!this.updateMissedProportions((int)instance.classValue())) return false;

        boolean label = false;
        double predictionValue = ClassifierUtils.combinePredictionsMax(predictionValues);

        switch (this.strategy) {
            case RANDOM:
                label = this.labelRandom(); break;
            case FIXED:
                label = this.labelFixed(predictionValue); break;
            case VARIABLE:
                label = this.labelVariable(predictionValue); break;
            case RAND_VARIABLE:
                label = this.labelRandomizedVariable(predictionValue); break;
            case SAMPLING:
                label = this.labelSelectiveSampling(predictionValue, instance); break;
            case CDDM:
                label = this.labelContinuousDDM(predictionValue, driftIndicators.get("error"), instance.numClasses()); break;
            case CEDDM:
                label = this.labelContinuousEDDM(predictionValue, driftIndicators.get("similarity"), instance.numClasses()); break;
            case FIRST:
                label = true; break;
            default:
                return label;
        }

        if (label) this.labeledInstances++;

        return label;
    }

    private boolean labelRandom() { return (this.random.nextDouble() < this.budget); }

    private boolean labelFixed(double predictionValue) { return (predictionValue <= this.fixedThreshold); }

    private boolean labelVariable(double predictionValue) {
        if (predictionValue <= this.variableThreshold) {
            this.variableThreshold *= (1 - this.variableThresholdStep);
            return true;
        }

        this.variableThreshold *= (1 + this.variableThresholdStep);
        return false;
    }

    private boolean labelRandomizedVariable(double predictionValue) {
        predictionValue = predictionValue / (this.random.nextGaussian() + 1.0);
        return labelVariable(predictionValue);
    }

    private boolean labelSelectiveSampling(double predictionValue, Instance instance) {
        double p = Math.abs(predictionValue - 1.0 / (instance.numClasses()));
        double budget = this.budget / (this.budget + p);

        return (this.random.nextDouble() <= budget);
    }

    private boolean labelContinuousDDM(double predictionValue, double meanError, int numClasses) {
        this.variableThreshold = Math.tanh(2 * (meanError + 1.0 / numClasses));
        return (predictionValue <= this.variableThreshold);
    }

    private boolean labelContinuousEDDM(double predictionValue, double similarity, int numClasses) {
        this.variableThreshold = 10 * (1.0 / numClasses - 1) * similarity + (10 - 9.0 / numClasses);
        return (predictionValue <= this.variableThreshold);
    }

    public UncertaintyStrategy setFixedThreshold(double fixedThreshold) {
        this.fixedThreshold = fixedThreshold;
        return this;
    }

    public UncertaintyStrategy setVariableThresholdStep(double variableThresholdStep) {
        this.variableThresholdStep = variableThresholdStep;
        return this;
    }

    @Override
    public HashMap<String, Double> getSeriesParameters(Instance instance, HashMap<String, Double> driftIndicators) {
        HashMap<String, Double> parameters = new HashMap<>(super.getSeriesParameters(instance, driftIndicators));
        this.trackableParameters.put("varThresh", this.variableThreshold);
        parameters.putAll(this.trackableParameters);
        return parameters;
    }

    @Override
    public ArrayList<String> getSeriesParameterNames() {
        ArrayList<String> parameterNames = new ArrayList<>(super.getSeriesParameterNames());
        parameterNames.add("varThresh");
        return parameterNames;
    }

    @Override
    public HashMap<String, Double> getAggregateParameters() {
        return super.getAggregateParameters();
    }

}
