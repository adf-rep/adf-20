package cls.df;

import cls.df.base.ARF;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import javafx.util.Pair;
import utils.ClassifierUtils;
import utils.Commons;
import utils.windows.WindowedValue;

import java.util.ArrayList;
import java.util.HashMap;

class InputLayer {

    private LayerSize size;
    private ArrayList<InputSubLayer> subLayers;
    private MultiGrainScanner inputScanner;

    private HashMap<String, WindowedValue> timeProfiler;

    InputLayer(LayerSize size, int stride, boolean rand, boolean adapt, boolean image) {
        this.size = size;
        this.inputScanner = new MultiGrainScanner(size.depth, stride, image);
        this.subLayers = new ArrayList<>();
        for (int i = 0; i < this.size.depth; i++) {
            this.subLayers.add(new InputSubLayer(size, rand ? 1 : 0, adapt));
        }

        this.timeProfiler = new HashMap<>();
        this.timeProfiler.put("grain_scan", new WindowedValue(1000));
        this.timeProfiler.put("input_transform", new WindowedValue(1000));
    }

    ArrayList<double[]> transform(Instance instance, boolean train) {
        long start = System.currentTimeMillis();
        ArrayList<Instances> multiGrainInstances = this.inputScanner.scanInstance(instance);
        if (train) this.timeProfiler.get("grain_scan").add(System.currentTimeMillis() - start);

        start = System.currentTimeMillis();
        ArrayList<double[]> multiGrainRepresentations = new ArrayList<>();
        for (int i = 0; i < this.size.depth; i++) {
            double[] grainRepresentation = this.subLayers.get(i).transform(multiGrainInstances.get(i), train);
            multiGrainRepresentations.add(grainRepresentation);
        }
        if (train) this.timeProfiler.get("input_transform").add(System.currentTimeMillis() - start);

        return multiGrainRepresentations;
    }

    HashMap<String, WindowedValue> getTimeProfiler() {
        return this.timeProfiler;
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

    class InputSubLayer {

        private ArrayList<ARF> forests;

        InputSubLayer(LayerSize size, int rand, boolean adapt) {
            this.forests = new ArrayList<>(DeepForest.createHoefddingForests(size, rand, adapt));
            this.forests.addAll(DeepForest.createRandomHoefddingForests(size, rand, adapt));
        }

        double[] transform(Instances grainInstances, boolean train) {
            double[] output = new double[0];

            for (int i = 0; i < grainInstances.size(); i++) {
                Instance grainInstance = grainInstances.get(i);
                for (ARF irf : this.forests) {
                    output = Commons.concat(output, ClassifierUtils.complementPrediction(irf.getVotesForInstance(grainInstance), grainInstance.numClasses()));
                }
            }

            if (train) {
                for (int i = 0; i < grainInstances.size(); i++) {
                    Instance grainInstance = grainInstances.get(i);
                    for (ARF irf : this.forests) {
                        irf.trainOnInstanceImpl(grainInstance);
                    }
                }
            }

            return output;
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
