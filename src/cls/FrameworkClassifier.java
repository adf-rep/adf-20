package cls;

import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;
import output.writer.OutputWriter;
import utils.Trackable;

import java.util.ArrayList;
import java.util.HashMap;

public abstract class FrameworkClassifier extends AbstractClassifier implements Trackable {

    abstract public void trainOnInstanceImpl(Instance instance, boolean labeled, HashMap<String, Double> indicators, int t);

    @Override
    public void trainOnInstanceImpl(Instance instance) {
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[0];
    }

    @Override
    public void getModelDescription(StringBuilder stringBuilder, int i) {
    }

    @Override
    public boolean isRandomizable() {
        return false;
    }

    @Override
    public HashMap<String, Double> getSeriesParameters(Instance instance, HashMap<String, Double> driftIndicators) {
        return new HashMap<>();
    }

    @Override
    public ArrayList<String> getSeriesParameterNames() {
        return new ArrayList<>();
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
