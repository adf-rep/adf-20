package cls.df;

import cls.df.base.ARF;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstanceImpl;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import javafx.util.Pair;
import moa.classifiers.core.driftdetection.ADWIN;
import moa.core.Utils;
import utils.ClassifierUtils;
import utils.Commons;
import utils.windows.WindowedValue;

import java.util.ArrayList;
import java.util.HashMap;

class CascadeLayer {

    private LayerSize size;
    private boolean append;
    private double weight;
    private ArrayList<CascadeSubLayer> subLayers;
    private boolean dynamicLayer;
    private int bestSubLayerIdx;

    private HashMap<String, WindowedValue> timeProfiler;

    CascadeLayer(LayerSize size, boolean rand, boolean adapt, boolean append, boolean dynamicLayer) {
        this.size = size;
        this.append = append;
        this.subLayers = new ArrayList<>();
        for (int i = 0; i < this.size.depth; i++) {
            this.subLayers.add(new CascadeSubLayer(size, rand ? 1 : 0, adapt));
        }
        this.dynamicLayer = dynamicLayer;
        this.bestSubLayerIdx = this.size.depth - 1;

        this.timeProfiler = new HashMap<>();
        this.timeProfiler.put("extending_input", new WindowedValue(1000));
        this.timeProfiler.put("cascade_transform", new WindowedValue(1000));
    }

    double[] transform(double[] layerInput, ArrayList<double[]> multiGrainRepresentations, Instance originalInput, boolean train) {
        double[] subLayerOutput = layerInput;
        double[] bestSubLayerOutput = new double[0];
        double bestSubLayerWeight = -Double.MIN_VALUE;
        boolean append = true;
        long extTotal = 0, cascTotal = 0;

        for (int i = 0; i < this.size.depth; i++) {
            double[] grainRepresentation = multiGrainRepresentations.get(i);
            long start = System.currentTimeMillis();
            Instance subLayerInput = CascadeLayer.extendInput(subLayerOutput, grainRepresentation, originalInput, append);
            extTotal += (System.currentTimeMillis() - start);

            start = System.currentTimeMillis();
            subLayerOutput = this.subLayers.get(i).transform(subLayerInput, train);
            cascTotal += (System.currentTimeMillis() - start);

            if (dynamicLayer) {
                double subLayerWeight = this.subLayers.get(i).getWeight();
                if (subLayerWeight > bestSubLayerWeight) {
                    bestSubLayerOutput = subLayerOutput.clone();
                    bestSubLayerWeight = subLayerWeight;
                    this.bestSubLayerIdx = i;
                }
            }

            append = this.append;
        }

        if (train) {
            this.weight = this.subLayers.get(bestSubLayerIdx).getWeight();
            this.timeProfiler.get("extending_input").add(extTotal);
            this.timeProfiler.get("cascade_transform").add(cascTotal);
        }

        return this.dynamicLayer ? bestSubLayerOutput : subLayerOutput;
    }

    static Instance extendInput(double[] input, double[] grainRepresentation, Instance instance, boolean append) {
        double[] extValues = Commons.concat(input, grainRepresentation);
        if (append) extValues = Commons.concat(extValues, instance.toDoubleArray());
        else extValues = Commons.concat(extValues, new double[] {instance.classValue()});

        Instance extRepresentation = new InstanceImpl(1.0, extValues);

        Attribute[] attributes = new Attribute[extValues.length];
        for (int i = 0; i < input.length; i++) {
            attributes[i] = new Attribute("prev_out_" + i);
        }
        for (int i = 0; i < grainRepresentation.length; i++) {
            attributes[i + input.length] = new Attribute("grain_rep_" + (i - input.length));
        }
        if (append) {
            for (int i = 0; i < instance.numAttributes(); i++) {
                attributes[i + input.length + grainRepresentation.length] = instance.attribute(i);
            }
        } else {
            attributes[input.length + grainRepresentation.length] = instance.classAttribute();
        }

        InstancesHeader header = new InstancesHeader();
        header.setAttributes(attributes);
        header.setClassIndex(extValues.length - 1);
        header.setRelationName("ext");
        extRepresentation.setDataset(header);

        return extRepresentation;
    }

    double getWeight() {
        return this.weight;
    }

    double[] getSubLayerWeights() {
        double[] subLayerWeights = new double[this.subLayers.size()];
        for (int i = 0; i < this.subLayers.size(); i++) {
            subLayerWeights[i] = this.subLayers.get(i).getWeight();
        }

        return subLayerWeights;
    }

    Pair<double[], double[]> getSubLayerDepthStats() {
        double[] subLayerDepthAverage = new double[this.subLayers.size()];
        double[] subLayerDepthMax = new double[this.subLayers.size()];

        for (int i = 0; i < this.subLayers.size(); i++) {
            Pair<Double, Integer> avgMax = this.subLayers.get(i).getDepthStats();
            subLayerDepthAverage[i] = avgMax.getKey();
            subLayerDepthMax[i] = avgMax.getValue();
        }

        return new Pair<>(subLayerDepthAverage, subLayerDepthMax);
    }

    double[] getOutputModelsWeights() {
        return this.subLayers.get(this.bestSubLayerIdx).getOutputModelsWeights();
    }

    HashMap<String, WindowedValue> getTimeProfiler() {
        return this.timeProfiler;
    }

    class CascadeSubLayer {

        private ArrayList<ARF> forests;
        private ADWIN weight;
        private ArrayList<ADWIN> subWeights;
        private double[] outputModelsWeights;

        CascadeSubLayer(LayerSize size, int rand, boolean adapt) {
            this.forests = new ArrayList<>(DeepForest.createHoefddingForests(size, rand, adapt));
            this.forests.addAll(DeepForest.createRandomHoefddingForests(size, rand, adapt));
            this.weight = new ADWIN();

            this.subWeights = new ArrayList<>();
            for (int i = 0; i < this.forests.size(); i++) {
                this.subWeights.add(new ADWIN());
            }
        }

        double[] transform(Instance subLayerInput, boolean train) {
            double[] output = new double[0];
            this.outputModelsWeights = this.getModelsWeights();

            for (int i = 0; i < this.forests.size(); i++) {
                double[] subOutput = ClassifierUtils.complementPrediction(this.forests.get(i).getVotesForInstance(subLayerInput), subLayerInput.numClasses());
                output = Commons.concat(output, subOutput);

                if (train) {
                    double correct = Utils.maxIndex(ClassifierUtils.prediction(subOutput, subLayerInput.numClasses())) == subLayerInput.classValue() ? 1.0 : 0.0;
                    this.subWeights.get(i).setInput(correct);

                    if (i == this.forests.size() - 1) {
                        int predIdx = Utils.maxIndex(ClassifierUtils.prediction(output, subLayerInput.numClasses(), this.outputModelsWeights));
                        this.weight.setInput(predIdx == subLayerInput.classValue() ? 1.0 : 0.0);
                    }

                    this.forests.get(i).trainOnInstanceImpl(subLayerInput);
                }
            }

            return output;
        }

        double getWeight() {
            double weight = this.weight.getEstimation();
            return Double.isNaN(weight) ? 0.0 : weight;
        }

        double[] getModelsWeights() {
            double[] outputWeights = new double[this.subWeights.size()];
            for (int i = 0; i < this.subWeights.size(); i++) {
                outputWeights[i] = this.subWeights.get(i).getEstimation();
            }
            return outputWeights;
        }

        double[] getOutputModelsWeights() {
            return this.outputModelsWeights;
        }

        Pair<Double, Integer> getDepthStats() {
            double sum = 0;
            int maxDepth = 0;

            for (int i = 0; i < this.forests.size(); i++) {
                Pair<Double, Integer> avgMax = this.forests.get(i).getDepthStats();
                sum += avgMax.getKey();

                int max = avgMax.getValue();
                if (max > maxDepth) {
                    maxDepth = max;
                }
            }

            return new Pair<>(sum / this.forests.size(), maxDepth);
        }

    }

}
