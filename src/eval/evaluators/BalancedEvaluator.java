package eval.evaluators;

import com.yahoo.labs.samoa.instances.Instance;
import eval.Evaluator;
import eval.experiment.ExperimentResult;
import eval.experiment.ExperimentRow;
import eval.experiment.ExperimentStream;
import framework.Framework;
import framework.FrameworkUpdate;
import moa.core.InstanceExample;
import moa.core.Utils;
import moa.evaluation.AdwinClassificationPerformanceEvaluator;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.evaluation.WindowClassificationPerformanceEvaluator;
import moa.streams.ArffFileStream;
import output.writer.OutputWriter;
import output.writer.PredictionsWriter;
import utils.eval.ResourcesMetrics;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.Map;

public class BalancedEvaluator implements Evaluator {

    public ExperimentResult evaluate(ExperimentRow experimentRow, ExperimentStream experimentStream) {
        System.out.print("\n" + new Date() + ": [" + experimentStream.streamName + " & " + experimentRow.label +
                (experimentRow.subLabel.isEmpty() ? "" : "#" + experimentRow.subLabel) + "] ");

        Framework framework = this.initializeFramework(experimentRow, experimentStream);
        ExperimentResult result = this.initializeResult(experimentRow);

        AdwinClassificationPerformanceEvaluator windowMeasurements = new AdwinClassificationPerformanceEvaluator();
        BasicClassificationPerformanceEvaluator averageMeasurements = new BasicClassificationPerformanceEvaluator();
        ResourcesMetrics rm = new ResourcesMetrics();

        int log = (int)(experimentStream.streamSize / 10.0);
        ArffFileStream stream = experimentStream.stream;
        int i = 0;

        while (stream.hasMoreInstances()) {
            if((i % log) == 0) System.out.print("#");
            //if ((i % (log / 10)) == 0) System.out.println(new Date() + ": Instance " + i);
            Instance instance = stream.nextInstance().getData();

            rm.startTimer();
            double[] votes = framework.classifier.getVotesForInstance(instance);
            rm.addClassificationTimeMeasurement(rm.stopTimer());

            averageMeasurements.addResult(new InstanceExample(instance), votes);
            windowMeasurements.addResult(new InstanceExample(instance), votes);

            if (framework.activeLearningStrategy == null || i >= framework.initInstances) {
                this.collectGeneralStepMetrics(result, windowMeasurements);
                this.collectFrameworkStepMetrics(result, framework, instance);
            }

            rm.startTimer();
            FrameworkUpdate update = framework.update(instance.copy(), votes, i);
            rm.addUpdateTimeMeasurement(rm.stopTimer());

            this.collectPrediction(result, framework, update, instance, votes);
            i++;
        }

        collectFinalMetrics(result, averageMeasurements, rm, framework, i);
        framework.ready = false;

        return result;
    }

    private ExperimentResult initializeResult(ExperimentRow row) {
        ExperimentResult result = new ExperimentResult(row.label, row.subLabel);
        result.seriesMeasurements.put(ExperimentResult.ACCURACY_SERIES, new ArrayList<>());
        result.seriesMeasurements.put(ExperimentResult.KAPPA_SERIES, new ArrayList<>());
        result.seriesMeasurements.put(ExperimentResult.KAPPA_T_SERIES, new ArrayList<>());

        if (row.framework.collectTrackableParameters) {
            for (String key : row.framework.getSeriesParameterNames()) {
                result.seriesMeasurements.put(key, new ArrayList<>());
            }
        }

        if (row.framework.collectOtherTrackable) {
            result.other.put(ExperimentResult.PREDICTIONS, new ArrayList<>());
            for (String key : row.framework.getOtherTrackableNames()) {
                result.other.put(key, new ArrayList<>());
            }
        }

        return result;
    }

    private Framework initializeFramework(ExperimentRow row, ExperimentStream stream) {
        Framework framework = row.framework.setInitInstances(
                row.framework.omitInit ? 0 : (int)(Framework.INIT_INSTANCES * stream.streamSize)
        );
        framework.classifier.setModelContext(stream.stream.getHeader());

        if (!framework.ready) {
            System.out.println("Framework is not prepared, you should reset it before using!");
        }

        return framework;
    }

    private void collectGeneralStepMetrics(ExperimentResult result, AdwinClassificationPerformanceEvaluator windowMeasurements) {
        result.seriesMeasurements.get(ExperimentResult.ACCURACY_SERIES).add(windowMeasurements.getFractionCorrectlyClassified());
        result.seriesMeasurements.get(ExperimentResult.KAPPA_SERIES).add(windowMeasurements.getKappaStatistic());
        result.seriesMeasurements.get(ExperimentResult.KAPPA_T_SERIES).add(windowMeasurements.getKappaTemporalStatistic());
    }

    private void collectFrameworkStepMetrics(ExperimentResult result, Framework framework, Instance instance) {
        if (framework.collectTrackableParameters) {
            for (Map.Entry<String, Double> entry : framework.getSeriesParameters(instance).entrySet()) {
                result.seriesMeasurements.get(entry.getKey()).add(entry.getValue());
            }

            for (Map.Entry<String, OutputWriter> entry : framework.getOtherTrackable().entrySet()) {
                result.other.get(entry.getKey()).add(entry.getValue());
            }
        }
    }

    private void collectPrediction(ExperimentResult result, Framework framework, FrameworkUpdate update, Instance instance, double[] votes) {
        if (framework.collectOtherTrackable) {
            Instance prediction = instance.copy();
            prediction.setClassValue(Utils.maxIndex(votes));
            update.predictedCorrect = (prediction.classValue() == instance.classValue());
            result.other.get(ExperimentResult.PREDICTIONS).add(new PredictionsWriter(prediction, update));
        }
    }

    private void collectFinalMetrics(ExperimentResult result, BasicClassificationPerformanceEvaluator averageMeasurements,
                                     ResourcesMetrics rm, Framework framework, int i) {
        result.averageMeasurements.put(ExperimentResult.ACCURACY, averageMeasurements.getFractionCorrectlyClassified());
        result.averageMeasurements.put(ExperimentResult.PRECISION, averageMeasurements.getPrecisionStatistic());
        result.averageMeasurements.put(ExperimentResult.RECALL, averageMeasurements.getRecallStatistic());
        result.averageMeasurements.put(ExperimentResult.F1, averageMeasurements.getF1Statistic());
        result.averageMeasurements.put(ExperimentResult.KAPPA, averageMeasurements.getKappaStatistic());
        result.averageMeasurements.put(ExperimentResult.KAPPA_T, averageMeasurements.getKappaTemporalStatistic());
        result.averageMeasurements.put(ExperimentResult.CLASSIFICATION_TIME, rm.getAvgClassificationTime());
        result.averageMeasurements.put(ExperimentResult.UPDATE_TIME, rm.getAvgUpdateTime());
        result.averageMeasurements.putAll(framework.getAggregateParameters());
        result.queriesFactor = (double) framework.activeLearningStrategy.labeledInstances / i;
        System.out.print(" Queried: " + result.queriesFactor);
    }

}
