package utils;

import com.yahoo.labs.samoa.instances.Instance;
import output.writer.OutputWriter;

import java.util.ArrayList;
import java.util.HashMap;

public interface Trackable {
    HashMap<String, Double> getSeriesParameters(Instance instance, HashMap<String, Double> driftIndicators);
    ArrayList<String> getSeriesParameterNames();
    HashMap<String, Double> getAggregateParameters();
    ArrayList<String> getOtherTrackableNames();
    HashMap<String, OutputWriter> getOtherTrackable();
}
