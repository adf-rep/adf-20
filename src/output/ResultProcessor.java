package output;

import eval.experiment.ExperimentResult;
import javafx.util.Pair;
import utils.drift.ConceptDrift;
import utils.drift.DriftingStreamDefinition;
import utils.FileUtils;
import utils.eval.RecoveryMetrics;

import java.io.File;
import java.util.*;
import java.util.stream.Collectors;

public class ResultProcessor {

    private String outputDir = "";
    private String title = "";
    private boolean groupable = false;

    public ResultProcessor(String outputDir, String streamName) {
        this.outputDir = outputDir + "/" + streamName;
        this.title = "Results for " + streamName;
    }

    public void writeResults(List<ExperimentResult> results, int logGranularity) {
        this.groupable = !results.get(0).subLabel.isEmpty();
        this.writePlots(results, logGranularity);
        this.writeRaw(results, logGranularity);
    }

    private void writePlots(List<ExperimentResult> results, int logGranularity) {
        BarPlotVisualizer barVis = new BarPlotVisualizer("Bar plot", this.title, this.outputDir + "/averages", this.groupable);
        barVis.plot(results, logGranularity, false);

        LinePlotVisualizer lineVis = new LinePlotVisualizer("Line plot", this.title, this.outputDir + "/series");
        if (this.groupable) {
            Map<String, List<ExperimentResult>> groupedResults = this.groupResultsByLabel(results);
            for (List<ExperimentResult> labelResults : groupedResults.values()) {
                lineVis.plot(labelResults, logGranularity, false);
            }
        }
        else {
            lineVis.plot(results, logGranularity, false);
        }
    }

    private void writeRaw(List<ExperimentResult> results, int logGranularity) {
        RaWritter.writeResultsToFile(results, this.outputDir, logGranularity);
    }

    private Map<String, List<ExperimentResult>> groupResultsByLabel(List<ExperimentResult> results) {
        return results.stream().collect(Collectors.groupingBy(w -> w.label));
    }

    private static void mergeResults(String rootDir, String[] resultPaths, String outputDir) {
        for (String resultDir : resultPaths) {
            File[] streamDirs = new File(rootDir + "/" + resultDir).listFiles(File::isDirectory);
            assert streamDirs != null;

            for (File streamDir : streamDirs) {
                File averagesDir = new File(streamDir.getAbsolutePath() + "/averages");
                List<String> measurementNames = FileUtils.getDirFileNames(averagesDir, ".data");

                for (String measurementName : measurementNames) {
                    String inputPath = streamDir.getPath() + "/averages/" + measurementName + ".data";
                    String outputPath = rootDir + "/" + outputDir + "/" + streamDir.getName() + "/averages/" + measurementName + ".data";

                    System.out.println("Appending " + inputPath + " to " + outputPath);
                    FileUtils.appendFile(inputPath, outputPath);
                }
            }

            for (File streamDir : streamDirs) {
                File averagesDir = new File(streamDir.getAbsolutePath() + "/series");
                List<String> measurementNames = FileUtils.getDirFileNames(averagesDir, ".data");

                for (String measurementName : measurementNames) {
                    String inputPath = streamDir.getPath() + "/series/" + measurementName + ".data";
                    String outputPath = rootDir + "/" + outputDir + "/" + streamDir.getName() + "/series/" + measurementName + ".data";

                    System.out.println("Appending " + inputPath + " to " + outputPath);
                    FileUtils.appendFile(inputPath, outputPath);
                }
            }
        }

        File[] outputStreamDirs = new File(rootDir + "/" + outputDir).listFiles(File::isDirectory);
        assert outputStreamDirs != null;

        for (File outputStreamDir : outputStreamDirs) {
            String streamName = outputStreamDir.getName();
            File averagesFile = new File(outputStreamDir.getPath() + "/averages");
            List<String> measurementNames = FileUtils.getDirFileNames(averagesFile, ".data");

            for (String measurementName : measurementNames) {
                String dataFilepath = averagesFile.getPath() + "/" + measurementName + ".data";
                BarPlotVisualizer barVis = new BarPlotVisualizer("Bar plot", "Results for " + streamName,
                        averagesFile.getPath(), false);

                System.out.println("Generating a bar chart for " + dataFilepath + " in " + averagesFile.getPath());
                barVis.plot(dataFilepath, measurementName, false);
            }
        }

        outputStreamDirs = new File(rootDir + "/" + outputDir).listFiles(File::isDirectory);
        assert outputStreamDirs != null;

        for (File outputStreamDir : outputStreamDirs) {
            String streamName = outputStreamDir.getName();
            File seriesFile = new File(outputStreamDir.getPath() + "/series");
            List<String> measurementNames = FileUtils.getDirFileNames(seriesFile, ".data");

            for (String measurementName : measurementNames) {
                String dataFilepath = seriesFile.getPath() + "/" + measurementName + ".data";
                LinePlotVisualizer lineVis = new LinePlotVisualizer("Line plot", "Results for " + streamName,
                        seriesFile.getPath());

                System.out.println("Generating a line plot for " + dataFilepath + " in " + seriesFile.getPath());
                lineVis.plot(dataFilepath, measurementName, false);
            }
        }
    }

    private static void generateOutputFromSummary(String rootDir, String outputDir, OutputType outputType) {
        File[] streamDirs = new File(rootDir + "/" + "cmp").listFiles(File::isDirectory);
        assert streamDirs != null;

        for (File streamDir : streamDirs) {
            List<String> measurementNames = FileUtils.getDirFileNames(streamDir , ".data");

            for (String measurementName : measurementNames) {
                String inputPath = streamDir.getPath() + "/" + measurementName + ".data";
                String outputPath;

                switch (outputType) {
                    case LATEX:
                        outputPath = rootDir + "/" + outputDir + "/" + streamDir.getName() + "/" + measurementName + "_latex.txt";
                        System.out.println("Generating LaTex table file for " + inputPath + " in " + outputPath);
                        FileUtils.parseRawResultToLatexTable(inputPath, outputPath);
                        break;
                    case CSV:
                        outputPath = rootDir + "/" + outputDir + "/" + measurementName + "_summary.csv";
                        System.out.println("Generating CSV file for " + inputPath + " in " + outputPath);
                        FileUtils.parseRawResultToCSV(inputPath, outputPath);
                        break;
                }
            }
        }
    }

    enum OutputType {LATEX, CSV}

    private static void generateRecoveryLowerBounds(String rootDir, ArrayList<DriftingStreamDefinition> definitions, String measure) {
        for (DriftingStreamDefinition definition : definitions) {
            LinkedHashMap<String, ArrayList<Double>> lowerBoundSeries = new LinkedHashMap<>();

            System.out.println("Loading " + definition.originStreamName + " for " + definition.streamName);
            File[] conceptsDirs = new File(rootDir + "/concepts_saturations/" + definition.originStreamName).listFiles(File::isDirectory);
            assert conceptsDirs != null;

            Pair<String[], LinkedHashMap<String, LinkedHashMap<String, Double>>> labelsConceptsSaturations = ResultProcessor.loadSaturations(conceptsDirs, measure);
            String[] algorithmLabels = labelsConceptsSaturations.getKey();
            LinkedHashMap<String, LinkedHashMap<String, Double>> conceptsSaturations = labelsConceptsSaturations.getValue();

            for (String algorithmLabel : algorithmLabels) {
                lowerBoundSeries.put(algorithmLabel, new ArrayList<>());
            }

            System.out.println("Calculating lower bounds...");
            int streamSize = definition.streamSize;
            int initInstancesNum = (int)(0.1 * definition.streamSize);
            ArrayList<ConceptDrift> drifts = definition.drifts;
            int currentDrift = 0;
            ConceptDrift cd = drifts.get(currentDrift);
            System.out.println(cd.concept + " " + cd.newConcept + " " + cd.p + " " + cd.width);

            for (int i = initInstancesNum; i < streamSize; i += definition.logGranularity) {
                if (currentDrift < drifts.size() - 1 && i > cd.p + ((drifts.get(currentDrift + 1).p - cd.p) / 2.0)) {
                    cd = drifts.get(++currentDrift);
                    System.out.println(cd.concept + " " + cd.newConcept + " " + cd.p + " " + cd.width);
                }

                LinkedHashMap<String, Double> conceptSaturations = conceptsSaturations.get(cd.concept);
                LinkedHashMap<String, Double> newConceptSaturations = conceptsSaturations.get(cd.newConcept);

                for (String algorithmLabel : algorithmLabels) {
                    double cAcc = conceptSaturations.get(algorithmLabel);
                    double ncAcc = newConceptSaturations.get(algorithmLabel);

                    double conceptProb = 1 - (1.0 / (1 + Math.exp((-4.0 / cd.width) * (i - cd.p))));
                    double lowerBound = Math.max(conceptProb * cAcc, ((1 - conceptProb) * ncAcc));

                    lowerBoundSeries.get(algorithmLabel).add(lowerBound);
                }
            }

            ArrayList<ExperimentResult> results = new ArrayList<>();
            for (Map.Entry<String, ArrayList<Double>> entry : lowerBoundSeries.entrySet()) {
                String[] algorithmLabel = entry.getKey().split("#");
                ExperimentResult result = new ExperimentResult(algorithmLabel[0], algorithmLabel[1]);
                result.seriesMeasurements.put(measure, entry.getValue());
                results.add(result);
            }

            System.out.println("Writing results for " + definition.streamName + "...");
            ResultProcessor resultProcessor = new ResultProcessor(rootDir, definition.streamName);
            resultProcessor.writeResults(results, 1);
        }
    }

    private static Pair<String[], LinkedHashMap<String, LinkedHashMap<String, Double>>> loadSaturations(File[] conceptsDirs, String measure) {
        String[] algorithmLabels = {};
        LinkedHashMap<String, LinkedHashMap<String, Double>> conceptsSaturations = new LinkedHashMap<>();

        for (File conceptDir : conceptsDirs) {
            String conceptSeriesPath = conceptDir.getAbsolutePath() + "/series/" + measure + ".data";
            System.out.println("Parsing " + conceptSeriesPath);

            LinkedHashMap<String, Double[]> conceptSeries = FileUtils.parseSeriesToHashMap(conceptSeriesPath);
            LinkedHashMap<String, Double> conceptAverages = new LinkedHashMap<>();
            algorithmLabels = conceptSeries.keySet().toArray(new String[conceptSeries.size()]);

            for (Map.Entry<String, Double[]> entry : conceptSeries.entrySet()) {
                Double[] values = entry.getValue();
                double avg = 0;

                for (int i = values.length / 2; i < values.length; i++) {
                    avg += values[i];
                }

                avg = avg / (values.length / 2);
                conceptAverages.put(entry.getKey(), avg);
            }

            conceptsSaturations.put(conceptDir.getName(), conceptAverages);
        }

        return new Pair<>(algorithmLabels, conceptsSaturations);
    }

    private static Pair<String[], LinkedHashMap<String, LinkedHashMap<String, ArrayList<Double>>>> loadResultSeries(File[] resultSeriesDir, String measure) {
        String[] algorithmLabels = {};
        LinkedHashMap<String, LinkedHashMap<String, ArrayList<Double>>> conceptResultSeries = new LinkedHashMap<>();

        for (File resultDir : resultSeriesDir) {
            String resultSeriesPath = resultDir.getAbsolutePath() + "/series/" + measure + ".data";
            System.out.println("Parsing " + resultSeriesPath);

            LinkedHashMap<String, Double[]> resultSeries = FileUtils.parseSeriesToHashMap(resultSeriesPath);
            LinkedHashMap<String, ArrayList<Double>> conceptAverages = new LinkedHashMap<>();
            algorithmLabels = resultSeries.keySet().toArray(new String[resultSeries.size()]);

            for (Map.Entry<String, Double[]> entry : resultSeries.entrySet()) {
                ArrayList<Double> values = new ArrayList<>();
                values.addAll(Arrays.asList(entry.getValue()));
                conceptAverages.put(entry.getKey(), values);
            }

            conceptResultSeries.put(resultDir.getName(), conceptAverages);
        }

        return new Pair<>(algorithmLabels, conceptResultSeries);
    }

//    private static void generateRecoveryMetrics(String rootDir, String resultsDir, String satsDir, HashMap<String, DriftingStreamDefinition> definitions, String measure) {
//        String resultsPath = rootDir + "/" + resultsDir;
//        File[] resultsFiles = new File(resultsPath).listFiles(File::isDirectory);
//        assert resultsFiles != null;
//
//        for (File resultsFile : resultsFiles) {
//            if (!definitions.containsKey(resultsFile.getName())) continue;
//
//            System.out.println("Processing results for " + resultsFile.getName());
//            String resultsSeriesPath = resultsFile.getAbsolutePath() + "/series/" + measure + ".data";
//            DriftingStreamDefinition definition = definitions.get(resultsFile.getName());
//
//            String conceptsPath = rootDir + "/" + satsDir + "/concepts_saturations/" + definition.originStreamName;
//            File[] conceptsDirs = new File(conceptsPath).listFiles(File::isDirectory);
//            assert conceptsDirs != null;
//
//            String lowerBoundsPath = rootDir + "/" + satsDir + "/lower_bounds/" + definition.originStreamName;
//            File[] lowerBoundDirs = new File(lowerBoundsPath).listFiles(File::isDirectory);
//            assert lowerBoundDirs != null;
//
//            System.out.println("Loading " + conceptsPath);
//            Pair<String[], LinkedHashMap<String, LinkedHashMap<String, Double>>> labelsConceptsSaturations = ResultProcessor.loadSaturations(conceptsDirs, measure);
//            String[] algorithmLabels = labelsConceptsSaturations.getKey();
//            LinkedHashMap<String, LinkedHashMap<String, Double>> conceptsSaturations = labelsConceptsSaturations.getValue();
//
//            System.out.println("Loading " + lowerBoundsPath);
//            Pair<String[], LinkedHashMap<String, LinkedHashMap<String, ArrayList<Double>>>> labelsLowerBoundSeries = ResultProcessor.loadResultSeries(lowerBoundDirs, measure);
//            LinkedHashMap<String, LinkedHashMap<String, ArrayList<Double>>> streamLowerBoundSeries = labelsLowerBoundSeries.getValue();
//
//            LinkedHashMap<String, RecoveryMetrics> recoveryMetrics = new LinkedHashMap<>();
//            for (String algorithmLabel : algorithmLabels) {
//                recoveryMetrics.put(algorithmLabel, new RecoveryMetrics(definition.drifts.size(), definition.streamSize));
//            }
//
//            System.out.println("Parsing " + resultsSeriesPath);
//            LinkedHashMap<String, Double[]> resultsSeries = FileUtils.parseSeriesToHashMap(resultsSeriesPath);
//
//            int initInstancesNum = (int)(0.1 * definition.streamSize);
//            ArrayList<ConceptDrift> drifts = definition.drifts;
//            int currentDrift = 0;
//            int driftAnalysisBoundary = (int)(drifts.get(currentDrift + 1).p - (drifts.get(currentDrift + 1).width / 2.0));
//            ConceptDrift cd = drifts.get(currentDrift);
//            boolean isDrift;
//
//            System.out.println(cd.concept + " " + cd.newConcept + " " + cd.p + " " + cd.width);
//
//            for (int i = initInstancesNum; i < definition.streamSize; i += definition.logGranularity) {
//                int j = (i - initInstancesNum) / definition.logGranularity;
//                int driftBoundary = (int)(cd.width >= 20000 ? cd.width / 2.0 : 10000);
//                isDrift = ((i >= cd.p - (cd.width / 2.0)) && (i <= cd.p + driftBoundary));
//
//                if (currentDrift < drifts.size() - 1 && i > driftAnalysisBoundary) {
//                    cd = drifts.get(++currentDrift);
//                    System.out.println(cd.concept + " " + cd.newConcept + " " + cd.p + " " + cd.width);
//
//                    if (currentDrift < drifts.size() - 1) {
//                        driftAnalysisBoundary = (int)(drifts.get(currentDrift + 1).p - (drifts.get(currentDrift + 1).width / 2.0));
//                    } else {
//                        driftAnalysisBoundary = definition.streamSize;
//                    }
//                }
//
//                LinkedHashMap<String, Double> conceptSaturations = conceptsSaturations.get(cd.concept);
//                LinkedHashMap<String, Double> newConceptSaturations = conceptsSaturations.get(cd.newConcept);
//                LinkedHashMap<String, ArrayList<Double>> lowerBoundSeries = streamLowerBoundSeries.get(definition.streamName);
//
//                for (String algorithmLabel : algorithmLabels) {
//                    double acc = resultsSeries.get(algorithmLabel)[j];
//                    double lowerBound = lowerBoundSeries.get(algorithmLabel).get(j);
//
//                    if (isDrift) {
//                        recoveryMetrics.get(algorithmLabel).updateAccDrifts(acc, currentDrift);
//                        recoveryMetrics.get(algorithmLabel).updateLowOptDrifts(lowerBound, currentDrift);
//                    } else {
//                        recoveryMetrics.get(algorithmLabel).updateAccStable(acc, currentDrift);
//                        recoveryMetrics.get(algorithmLabel).updateLowOptStable(lowerBound, currentDrift);
//                    }
//
//                    if (i > cd.p - (cd.width / 2.0)) {
//                        double cAcc = conceptSaturations.get(algorithmLabel);
//                        double ncAcc = newConceptSaturations.get(algorithmLabel);
//
//                        double loss = (Math.min(cAcc, ncAcc) - acc) /  Math.min(cAcc, ncAcc);
//                        recoveryMetrics.get(algorithmLabel).updateLoss(loss, currentDrift);
//                    }
//                }
//            }
//
//            ArrayList<ExperimentResult> results = new ArrayList<>();
//            for (Map.Entry<String, RecoveryMetrics> entry : recoveryMetrics.entrySet()) {
//                String[] algorithmLabel = entry.getKey().split("#");
//
//                ExperimentResult result = new ExperimentResult(algorithmLabel[0], algorithmLabel[1]);
//                result.averageMeasurements.put("max_loss", entry.getValue().getAverageMaxLoss());
//
//                result.averageMeasurements.put("drifts", entry.getValue().getAverageAccDrifts());
//                result.averageMeasurements.put("stable", entry.getValue().getAverageAccStable());
//                result.averageMeasurements.put("std_drifts", entry.getValue().getStdAccDrifts());
//                result.averageMeasurements.put("std_stable", entry.getValue().getStdAccStable());
//
//                result.averageMeasurements.put("opt_drifts", entry.getValue().getAverageLowOptDrifts());
//                result.averageMeasurements.put("opt_stable", entry.getValue().getAverageLowOptStable());
//
//                result.averageMeasurements.put("diff_drifts", entry.getValue().getAverageLowOptDrifts() - entry.getValue().getAverageAccDrifts());
//                result.averageMeasurements.put("diff_stable", entry.getValue().getAverageLowOptStable() - entry.getValue().getAverageAccStable());
//
//                results.add(result);
//            }
//
//            System.out.println("Writing results for " + definition.streamName + "...");
//            ResultProcessor resultProcessor = new ResultProcessor(rootDir, definition.streamName);
//            resultProcessor.writeResults(results, 1);
//        }
//    }

    public static void generateStreamStateStatistics(String rootDir, String resultsDir, HashMap<String, DriftingStreamDefinition> definitions, String measure) {
        String resultsPath = rootDir + "/" + resultsDir;
        File[] resultsFiles = new File(resultsPath).listFiles(File::isDirectory);
        assert resultsFiles != null;

        for (File resultsFile : resultsFiles) {
            if (!definitions.containsKey(resultsFile.getName())) continue;

            System.out.println("Processing results for " + resultsFile.getName());
            String resultsSeriesPath = resultsFile.getAbsolutePath() + "/series/" + measure + ".data";
            DriftingStreamDefinition definition = definitions.get(resultsFile.getName());

            System.out.println("Parsing " + resultsSeriesPath);
            LinkedHashMap<String, Double[]> resultsSeries = FileUtils.parseSeriesToHashMap(resultsSeriesPath);

            String[] algorithmLabels = resultsSeries.keySet().toArray(new String[0]);

            LinkedHashMap<String, RecoveryMetrics> recoveryMetrics = new LinkedHashMap<>();
            for (String algorithmLabel : algorithmLabels) {
                recoveryMetrics.put(algorithmLabel, new RecoveryMetrics(definition.drifts.size(), definition.streamSize));
            }

            int initInstancesNum = (int)(0.05 * definition.streamSize);
            ArrayList<ConceptDrift> drifts = definition.drifts;
            int currentDrift = 0;
            int driftAnalysisBoundary = (int)(drifts.get(currentDrift + 1).p - (drifts.get(currentDrift + 1).width / 2.0));
            ConceptDrift cd = drifts.get(currentDrift);
            boolean isDrift;

            System.out.println(cd.concept + " " + cd.newConcept + " " + cd.p + " " + cd.width);

            for (int i = initInstancesNum; i < definition.streamSize; i += definition.logGranularity) {
                int j = (i - initInstancesNum) / definition.logGranularity;
                int driftBoundary = (int)(cd.width >= 20000 ? cd.width / 2.0 : 10000);
                isDrift = ((i >= cd.p - (cd.width / 2.0)) && (i <= cd.p + driftBoundary));

                if (currentDrift < drifts.size() - 1 && i > driftAnalysisBoundary) {
                    cd = drifts.get(++currentDrift);
                    System.out.println(cd.concept + " " + cd.newConcept + " " + cd.p + " " + cd.width);

                    if (currentDrift < drifts.size() - 1) {
                        driftAnalysisBoundary = (int)(drifts.get(currentDrift + 1).p - (drifts.get(currentDrift + 1).width / 2.0));
                    } else {
                        driftAnalysisBoundary = definition.streamSize;
                    }
                }

                for (String algorithmLabel : algorithmLabels) {
                    double acc = resultsSeries.get(algorithmLabel)[j];

                    if (isDrift) {
                        recoveryMetrics.get(algorithmLabel).updateAccDrifts(acc, currentDrift);
                    } else {
                        recoveryMetrics.get(algorithmLabel).updateAccStable(acc, currentDrift);
                    }
                }
            }

            ArrayList<ExperimentResult> results = new ArrayList<>();
            for (Map.Entry<String, RecoveryMetrics> entry : recoveryMetrics.entrySet()) {
                String[] algorithmLabel = entry.getKey().split("#");

                ExperimentResult result = new ExperimentResult(algorithmLabel[0], algorithmLabel[1]);

                result.averageMeasurements.put("drifts", entry.getValue().getAverageAccDrifts());
                result.averageMeasurements.put("stable", entry.getValue().getAverageAccStable());
                //result.averageMeasurements.put("std_drifts", entry.getValue().getStdAccDrifts());
                //result.averageMeasurements.put("std_stable", entry.getValue().getStdAccStable());

                results.add(result);
            }

            System.out.println("Writing results for " + definition.streamName + "...");
            ResultProcessor resultProcessor = new ResultProcessor(rootDir, definition.streamName);
            resultProcessor.writeResults(results, 1);
        }
    }

    public static void concatResults(String rootDir, String resultsDir, String outputDir) {
        String resultsPath = rootDir + "/" + resultsDir;
        File[] resultsFiles = new File(resultsPath).listFiles(File::isDirectory);
        assert resultsFiles != null;

        LinkedHashMap<String, LinkedHashMap<String, LinkedHashMap<String, Double>>> streamsResults = new LinkedHashMap<>(); // stream-measure-algorithm

        for (File resultsFile : resultsFiles) {
            System.out.println("Processing results for " + resultsFile.getName());
            String averageResltsDir = resultsFile.getAbsolutePath() + "/averages";
            List<String> measurementNames = FileUtils.getDirFileNames(new File(averageResltsDir), ".data");

            LinkedHashMap<String, LinkedHashMap<String, Double>> measurementsAverages = new LinkedHashMap<>();

            for (String measurementName : measurementNames) {
                String measurementPath = averageResltsDir + "/" + measurementName + ".data";
                LinkedHashMap<String, Double> algorithmsAverages = FileUtils.parseAveragesToHashMap(measurementPath);
                measurementsAverages.put(measurementName, algorithmsAverages);
            }

            streamsResults.put(resultsFile.getName(), measurementsAverages);
        }

        String outputPath = rootDir + "/" + outputDir + "/all.csv";
        RaWritter.writeSMAResultsToCSV(outputPath, streamsResults);
    }

    public static void main(String[] args) {
//        ResultProcessor.mergeResults("../Results/os/imbalanced/iadem2",
//                new String[]{"al_rand", "al_randvar", "se", "pw_uni", "pw_sigm", "smote_fix_uni", "smote_fix_poiss2"}, "cmp2");
//        ResultProcessor.generateOutputFromSummary("../Results/os/imbalanced/iadem2", "summary_csv2", OutputType.CSV);

//        ResultProcessor.generateRecoveryLowerBounds("../Results/div/aht2/synth_sat_b10",
//                DriftingStreamDefinition.createDriftDefinitions(), ExperimentResult.ACCURACY_SERIES);
//        ResultProcessor.generateRecoveryMetrics("../Results/div/aht2", "synth_b10", "synth_sat_b10/accuracy",
//                DriftingStreamDefinition.createDriftDefinitionsMap(), ExperimentResult.ACCURACY_SERIES);
//        ResultProcessor.concatResults("../Results/div/aht2", "synth_b10", "synth_b10");
//        ResultProcessor.concatResults("../Results/div/aht", "real_sensor_b10/rest", "real_sensor_b10");

        ResultProcessor.mergeResults("/home/lk/Results/os/det/aht",
                new String[]{"0.01", "0.05", "0.1", "0.2", "0.5"}, "cmp");
        //ResultProcessor.concatResults("/home/lk/Results/os/aht/gens-adapt/state/0.1", "", "summary");
//        ResultProcessor.generateStreamStateStatistics("/home/lk/Results/os/aht", "gens-adapt/general/0.1",
//                DriftingStreamDefinition.createDriftDefinitionsMap(), ExperimentResult.ACCURACY_SERIES);

        System.exit(0);
    }
}
