package strategies.al;

import com.yahoo.labs.samoa.instances.Instance;
import output.writer.OutputWriter;
import utils.math.MathUtils;
import utils.Trackable;
import utils.windows.WindowedValue;

import java.util.*;

public abstract class ActiveLearningStrategy implements Trackable {

    protected double budget = 1.0;
    public int labeledInstances = 0;
    int allInstances = 0;
    double currentCost = 0.0;
    int iterationCount = 0;

    protected int windowSize;
    HashMap<Integer, WindowedValue> labeledClassProportions = new HashMap<>();
    HashMap<Integer, WindowedValue> missedClassProportions = new HashMap<>();
    private boolean binary = false;
    private boolean collectProportions = false;

    private HashMap<String, Double> seriesParameters = new HashMap<>();
    protected Random random = new Random();

    abstract public boolean queryLabel(Instance instance, double[] predictionValues, HashMap<String, Double> driftIndicators);

    public void reset() {
        this.labeledInstances = 0;
        this.allInstances = 0;
        this.currentCost = 0.0;
        this.iterationCount = 0;
        this.seriesParameters.clear();
    }

    public void update(Instance instance, boolean label) {
        if (this.collectProportions && label) {
            this.updateLabeledProportions(instance, (int)instance.classValue());
        }
    }

    private void updateLabeledProportions(Instance instance, int classLabel) {
        for (int i = 0; i < instance.numClasses(); i++) {
            if (!labeledClassProportions.containsKey(i)) {
                labeledClassProportions.put(i, new WindowedValue(this.windowSize));
            }

            labeledClassProportions.get(i).add((i == classLabel) ? 1.0 : 0.0);
        }
    }

    boolean updateMissedProportions(int classLabel) {
        if (!missedClassProportions.containsKey(classLabel)) {
            missedClassProportions.put(classLabel, new WindowedValue(Integer.MAX_VALUE));
        }

        if (this.currentCost >= this.budget) {
            missedClassProportions.get(classLabel).add(1.0);
            return false;
        } else {
            missedClassProportions.get(classLabel).add(0.0);
        }

        return true;
    }

    private double getLcpGMean(int classesNum) {
        if (classesNum == 0) return 0;
        double lcp_gm = 1.0;

        for (int i = 0; i < classesNum; i++) {
            lcp_gm *= this.labeledClassProportions.get(i).getAverage();
        }

        return MathUtils.root(lcp_gm, classesNum);
    }

    public ActiveLearningStrategy isBinary() {
        this.binary = true;
        return this;
    }

    public ActiveLearningStrategy setWindowSize(int windowSize) {
        this.windowSize = windowSize;
        return this;
    }

    public ActiveLearningStrategy setCollectProportions(boolean collectProportions) {
        this.collectProportions = collectProportions;
        return this;
    }

    @Override
    public HashMap<String, Double> getSeriesParameters(Instance instance, HashMap<String, Double> driftIndicators) {
        this.seriesParameters.put("budget", (double) this.labeledInstances / this.allInstances);
        if (this.collectProportions) {
            this.seriesParameters.put("lcp_gm", this.getLcpGMean(instance != null ? instance.numClasses() : 0));

            if (this.binary) {
                this.seriesParameters.put("lcp_c1",
                        labeledClassProportions.containsKey(0) ? labeledClassProportions.get(0).getAverage() : 0.0);
                this.seriesParameters.put("lcp_c2",
                        labeledClassProportions.containsKey(1) ? labeledClassProportions.get(1).getAverage() : 0.0);
                this.seriesParameters.put("mcp_c1",
                        missedClassProportions.containsKey(0) ? missedClassProportions.get(0).getAverage() : 0.0);
                this.seriesParameters.put("mcp_c2",
                        missedClassProportions.containsKey(1) ? missedClassProportions.get(1).getAverage() : 0.0);
            }
        }

        return this.seriesParameters;
    }

    @Override
    public ArrayList<String> getSeriesParameterNames() {
        ArrayList<String> parameterNames = new ArrayList<>(Collections.singletonList("budget"));
        if (this.collectProportions) {
            parameterNames.add("lcp_gm");
            if (binary) parameterNames.addAll(Arrays.asList("lcp_c1", "lcp_c2", "mcp_c1", "mcp_c2"));
        }
        return parameterNames;
    }

    @Override
    public HashMap<String, Double> getAggregateParameters() {
        return new HashMap<>();
    }

    @Override
    public ArrayList<String> getOtherTrackableNames() {
        return new ArrayList<>();
    }

    @Override
    public HashMap<String, OutputWriter> getOtherTrackable() {
        return new HashMap<>();
    }

}
