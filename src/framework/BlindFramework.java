package framework;

import output.writer.OutputWriter;
import strategies.al.ActiveLearningStrategy;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.Classifier;
import utils.Trackable;
import utils.TrackableFramework;

import java.util.ArrayList;
import java.util.HashMap;

public class BlindFramework extends Framework implements TrackableFramework {

    public BlindFramework(Classifier classifier, ActiveLearningStrategy activeLearningStrategy) {
        this.classifier = classifier;
        this.classifier.prepareForUse();
        this.activeLearningStrategy = activeLearningStrategy;
    }

    @Override
    public void reset() {
        this.classifier.resetLearning();
        this.activeLearningStrategy.reset();
        this.ready = true;
    }

    @Override
    public FrameworkUpdate update(Instance instance, double[] votes, int n) {
        if (!this.omitInit && n <= this.initInstances) {
            Framework.trainClassifier(this.classifier, instance, true, null, n);
            this.activeLearningStrategy.update(instance, true);

            return new FrameworkUpdate(true);
        }

        boolean learnActively = this.activeLearningStrategy.queryLabel(instance, votes, null);
        this.activeLearningStrategy.update(instance, learnActively);

        if (learnActively) {
            Framework.trainClassifier(this.classifier, instance, true, null, n);
        } else {
            Framework.trainClassifier(this.classifier, instance, false, null, n);
        }

        return new FrameworkUpdate(learnActively);
    }

    @Override
    public HashMap<String, Double> getSeriesParameters(Instance instance) {
        if (!this.collectTrackableParameters) return null;
        else {
            HashMap<String, Double> parameters = new HashMap<>();
            if (this.classifier instanceof Trackable) {
                parameters.putAll(((Trackable) this.classifier).getSeriesParameters(instance, null));
            }

            parameters.putAll(this.activeLearningStrategy.getSeriesParameters(instance, null));

            return parameters;
        }
    }

    @Override
    public ArrayList<String> getSeriesParameterNames() {
        if (!this.collectTrackableParameters) return null;
        else {
            ArrayList<String> parameterNames = new ArrayList<>();
            if (this.classifier instanceof Trackable) {
                parameterNames.addAll(((Trackable) this.classifier).getSeriesParameterNames());
            }
            parameterNames.addAll(this.activeLearningStrategy.getSeriesParameterNames());

            return parameterNames;
        }
    }

    @Override
    public HashMap<String, Double> getAggregateParameters() {
        HashMap<String, Double> parameters = new HashMap<>();
        if (!this.collectTrackableParameters) return parameters;

        if (this.classifier instanceof Trackable) {
            parameters.putAll(((Trackable) this.classifier).getAggregateParameters());
        }
        parameters.putAll(this.activeLearningStrategy.getAggregateParameters());

        return parameters;
    }

    @Override
    public ArrayList<String> getOtherTrackableNames() {
        if (this.collectOtherTrackable && this.classifier instanceof Trackable) {
            return ((Trackable) this.classifier).getOtherTrackableNames();
        }
        return new ArrayList<>();
    }

    @Override
    public HashMap<String, OutputWriter> getOtherTrackable() {
        if (this.collectOtherTrackable && this.classifier instanceof Trackable) {
            return ((Trackable) this.classifier).getOtherTrackable();
        }
        return new HashMap<>();
    }

}
