package cls.df;

import cls.FrameworkClassifier;
import cls.df.base.ARF;
import cls.df.base.ARFHT;
import com.yahoo.labs.samoa.instances.Instance;
import javafx.util.Pair;
import moa.classifiers.trees.ARFHoeffdingTree;
import moa.options.ClassOption;
import utils.ClassifierUtils;
import utils.windows.WindowedValue;

import java.util.*;

public class DeepForest extends FrameworkClassifier {

    private LayerSize inputSize, cascadeSize;
    private boolean grainScan;
    private int stride;
    private boolean rand;
    private boolean adapt;
    private boolean append;
    private boolean image;
    private boolean dynamicLayer;

    private InputLayer inputLayer;
    private ArrayList<CascadeLayer> cascadeLayers;

    private boolean performanceProfiler;

    public DeepForest(LayerSize inputSize, LayerSize cascadeSize, int stride, boolean rand, boolean adapt, boolean append, boolean image) {
        this.grainScan = true;
        this.inputSize = inputSize;
        this.cascadeSize = cascadeSize;
        this.stride = stride;
        this.rand = rand;
        this.adapt = adapt;
        this.append = append;
        this.image = image;
        this.dynamicLayer = true;
        this.performanceProfiler = false;

        this.resetLearningImpl();
    }

    public DeepForest(LayerSize cascadeSize, boolean rand, boolean adapt, boolean append, boolean dynamicLayer) {
        this.grainScan = false;
        this.cascadeSize = cascadeSize;
        this.rand = rand;
        this.adapt = adapt;
        this.append = append;
        this.dynamicLayer = dynamicLayer;
        this.performanceProfiler = false;

        this.resetLearningImpl();
    }


    @Override
    public void resetLearningImpl() {
        if (grainScan) this.inputLayer = new InputLayer(this.inputSize, this.stride, this.rand, this.adapt, this.image);
        this.cascadeLayers = new ArrayList<>();
        this.cascadeLayers.add(new CascadeLayer(this.cascadeSize, this.rand, this.adapt, this.append, this.dynamicLayer));
    }

    @Override
    public void trainOnInstanceImpl(Instance instance, boolean labeled, HashMap<String, Double> indicators, int t) {
        this.feed(instance, true);
    }

    @Override
    public double[] getVotesForInstance(Instance instance) {
        return this.feed(instance, false);
    }

    private double[] feed(Instance instance, boolean train) {
        ArrayList<double[]> multiGrainRepresentations = this.grainScan ? this.inputLayer.transform(instance, train) : this.bypassGrainInstances();
        double[] output = new double[0];

        for (CascadeLayer cascadeLayer : this.cascadeLayers) {
            output = cascadeLayer.transform(output, multiGrainRepresentations, instance, train);
        }

        return ClassifierUtils.prediction(output, instance.numClasses(), this.lastLayer().getOutputModelsWeights());
    }

    private CascadeLayer lastLayer() {
        return this.cascadeLayers.get(this.cascadeLayers.size() - 1);
    }

    static ArrayList<ARF> createHoefddingForests(LayerSize size, int rand, boolean adapt) {
        ArrayList<ARF> hoeffdingForests = new ArrayList<>();

        for (int i = 0; i < size.numForests / Math.pow(2.0, rand); i++) {
            ARF arf = new ARF();
            arf.ensembleSizeOption.setValue(size.numTrees);
            if (!adapt) arf.disableDriftDetectionOption.set();
            arf.prepareForUse();
            hoeffdingForests.add(arf);
        }

        return hoeffdingForests;
    }

    static ArrayList<ARF> createRandomHoefddingForests(LayerSize size, int rand, boolean adapt) {
        ArrayList<ARF> randomHoeffdingForests = new ArrayList<>();

        for (int i = 0; i < rand * 0.5 * size.numForests; i++) {
            ARF rrf = new ARF();
            rrf.ensembleSizeOption.setValue(size.numTrees);
            if (!adapt) rrf.disableDriftDetectionOption.set();
            rrf.treeLearnerOption = new ClassOption("treeLearner", 'l', "Random Forest Tree.", ARFHT.class, "ARFHT -e 2000000 -g 50 -c 0.01 -l MC");
            rrf.prepareForUse();
            randomHoeffdingForests.add(rrf);
        }

        return randomHoeffdingForests;
    }

    private ArrayList<double[]> bypassGrainInstances() {
        ArrayList<double[]> bypass = new ArrayList<>();
        for (int i = 0; i < this.cascadeSize.depth; i++) {
            bypass.add(new double[0]);
        }

        return bypass;
    }

    public DeepForest setPerformanceProfiler(boolean set) {
        this.performanceProfiler = set;
        return this;
    }

    @Override
    public ArrayList<String> getSeriesParameterNames() {
        ArrayList<String> parameterNames = new ArrayList<>();

        if (this.performanceProfiler) {
            for (int i = 0; i < this.cascadeLayers.size(); i++) {
                for (int j = 0; j < this.cascadeLayers.get(i).getSubLayerWeights().length; j++) {
                    parameterNames.add("sublayerWeight_" + i + "_" + j);
                    parameterNames.add("sublayerAverageDepth_" + i + "_" + j);
                    parameterNames.add("sublayerMaxDepth_" + i + "_" + j);
                }
            }

            if (this.grainScan) {
                Pair<double[], double[]> inputLayerDepthStats = this.inputLayer.getSubLayerDepthStats();
                int ins = inputLayerDepthStats.getKey().length;
                for (int j = 0; j < ins; j++) {
                    parameterNames.add("inputLayerAverageDepth_" + j);
                    parameterNames.add("inputLayerMaxDepth_" + j);
                }
            }
        }

        return parameterNames;
    }

    @Override
    public HashMap<String, Double> getSeriesParameters(Instance instance, HashMap<String, Double> driftIndicators) {
        HashMap<String, Double> parameters = new HashMap<>();
        if (this.performanceProfiler) {
            for (int i = 0; i < this.cascadeLayers.size(); i++) {
                double[] subLayerWeights = this.cascadeLayers.get(i).getSubLayerWeights();
                Pair<double[], double[]> subLayerDepthStats = this.cascadeLayers.get(i).getSubLayerDepthStats();
                for (int j = 0; j < subLayerWeights.length; j++) {
                    parameters.put("sublayerWeight_" + i + "_" + j, subLayerWeights[j]);
                    parameters.put("sublayerAverageDepth_" + i + "_" + j, subLayerDepthStats.getKey()[j]);
                    parameters.put("sublayerMaxDepth_" + i + "_" + j, subLayerDepthStats.getValue()[j]);
                }
            }

            if (this.grainScan) {
                Pair<double[], double[]> inputLayerDepthStats = this.inputLayer.getSubLayerDepthStats();
                int ins = inputLayerDepthStats.getKey().length;
                for (int j = 0; j < ins; j++) {
                    parameters.put("inputLayerAverageDepth_" + j, inputLayerDepthStats.getKey()[j]);
                    parameters.put("inputLayerMaxDepth_" + j, inputLayerDepthStats.getValue()[j]);
                }
            }
        }

        return parameters;
    }

    @Override
    public HashMap<String, Double> getAggregateParameters() {
        HashMap<String, Double> parameters = new HashMap<>();
        if (this.performanceProfiler) {
            if (this.grainScan) {
                for (Map.Entry<String, WindowedValue> profiler : this.inputLayer.getTimeProfiler().entrySet()) {
                    parameters.put(profiler.getKey(), profiler.getValue().getAverage());
                }
            }

            for (Map.Entry<String, WindowedValue> profiler : this.cascadeLayers.get(0).getTimeProfiler().entrySet()) {
                parameters.put(profiler.getKey(), profiler.getValue().getAverage());
            }
        }

        return parameters;
    }

}
