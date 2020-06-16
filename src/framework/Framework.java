package framework;

import cls.FrameworkClassifier;
import output.writer.OutputWriter;
import strategies.al.ActiveLearningStrategy;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.Classifier;
import utils.TrackableFramework;

import java.util.ArrayList;
import java.util.HashMap;

abstract public class Framework implements TrackableFramework {

    public Classifier classifier;
    public ActiveLearningStrategy activeLearningStrategy;
    public boolean ready = true;
    public int initInstances;
    public boolean omitInit = false;
    public boolean collectTrackableParameters = true;
    public boolean collectOtherTrackable = false;
    public long allocatedMemory = -1;
    static public double INIT_INSTANCES = 0.05;

    abstract public void reset();
    abstract public FrameworkUpdate update(Instance instance, double[] votes, int i);
    abstract public HashMap<String, Double> getSeriesParameters(Instance instance);

    public Framework setInitInstances(int initInstancesNum) {
        this.initInstances = initInstancesNum;
        return this;
    }

    public Framework setOmitInit(boolean omitInit) {
        this.omitInit = omitInit;
        return this;
    }

    public Framework setCollectTrackableParameters(boolean collectTrackableParameters) {
        this.collectTrackableParameters = collectTrackableParameters;
        return this;
    }

    public Framework setCollectOtherTrackable(boolean collectOtherTrackable) {
        this.collectOtherTrackable = collectOtherTrackable;
        return this;
    }

    public static void trainClassifier(Classifier classifier, Instance instance, boolean labeled, HashMap<String, Double> indicators, int t) {
        if (classifier instanceof FrameworkClassifier) {
            ((FrameworkClassifier)classifier).trainOnInstanceImpl(instance, labeled, indicators, t);
        } else if (labeled) {
            classifier.trainOnInstance(instance);
        }
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
