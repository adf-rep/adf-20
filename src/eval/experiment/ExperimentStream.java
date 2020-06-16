package eval.experiment;
import moa.streams.ArffFileStream;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ExperimentStream {

    public ArffFileStream stream;
    public String streamName;
    public double streamSize;
    public int logGranularity;
    public ArrayList<Double> classRatios;

    public ExperimentStream(ArffFileStream stream, String streamName, double streamSize, int logGranularity) {
        this.stream = stream;
        this.streamName = streamName;
        this.streamSize = streamSize;
        this.logGranularity = logGranularity;
    }

    public ExperimentStream(ArffFileStream stream, String streamName, double streamSize, int logGranularity, ArrayList<Double> classRatios) {
        this.stream = stream;
        this.streamName = streamName;
        this.streamSize = streamSize;
        this.logGranularity = logGranularity;
        this.classRatios = classRatios;
    }

    public static List<ExperimentStream> createExperimentStreams(String rootDataDir) {
        List<ExperimentStream> streams = new ArrayList<>();

        // final
        streams.addAll(createRealDriftDeepStreams(rootDataDir + "/real-dl"));
        //streams.addAll(createRealStreams(rootDataDir + "/real"));

        // settings
        //streams.addAll(createRealDeepStreams(rootDataDir + "/real-dl"));

        return streams;
    }

    private static List<ExperimentStream> createRealStreams(String rootDataDir) {
        List<ExperimentStream> streams = new ArrayList<>();

        // shallow, d = 4
        streams.add(new ExperimentStream(new ArffFileStream(rootDataDir + "/ACTIVITY/TRANSFORMED/ACTIVITY.arff", 44), "ACTIVITY", 10853, 100));
        streams.add(new ExperimentStream(new ArffFileStream(rootDataDir + "/COVERTYPE/COVERTYPE.arff", 55), "COVERTYPE", 581012, 2500));
        streams.add(new ExperimentStream(new ArffFileStream(rootDataDir + "/GAS/GAS.arff", 129), "GAS", 13910, 100));
        streams.add(new ExperimentStream(new ArffFileStream(rootDataDir + "/SPAM/SPAM09/SPAM.arff", 500), "SPAM", 9324, 100));
        streams.add(new ExperimentStream(new ArffFileStream(rootDataDir + "/CONNECT4/CONNECT4.arff", 43), "CONNECT4", 67557, 500));

        // shallow, d = 2
        streams.add(new ExperimentStream(new ArffFileStream(rootDataDir + "/EEG/EEG.arff", 15), "EEG", 14980, 100));
        streams.add(new ExperimentStream(new ArffFileStream(rootDataDir + "/ELEC/ELEC.arff", 9), "ELEC", 45312, 100));
        streams.add(new ExperimentStream(new ArffFileStream(rootDataDir + "/WEATHER/WEATHER.arff", 9), "WEATHER", 18158, 200));
        streams.add(new ExperimentStream(new ArffFileStream(rootDataDir + "/POKER/POKER.arff", 11), "POKER", 829201, 2500));

        return streams;
    }

    private static List<ExperimentStream> createRealDriftDeepStreams(String rootDataDir) {
        List<ExperimentStream> streams = new ArrayList<>();

        // deep-1D, d = 5
        streams.add(new ExperimentStream(new ArffFileStream(rootDataDir + "/drift/text/BBC-D1.arff", 1001), "BBC-D1", 66750, 100));
        streams.add(new ExperimentStream(new ArffFileStream(rootDataDir + "/drift/text/AGNEWS-D1.arff", 170), "AGNEWS-D1", 60000, 100));
        streams.add(new ExperimentStream(new ArffFileStream(rootDataDir + "/drift/generic/SEMG-D1.arff", 3001), "SEMG-D1", 54000, 100));
        streams.add(new ExperimentStream(new ArffFileStream(rootDataDir + "/drift/text/SOGOU-D1.arff", 1501), "SOGOU-D1", 60000, 100));

        // deep-2D, d = 4
        streams.add(new ExperimentStream(new ArffFileStream(rootDataDir + "/drift/vis/CMATER-BANGLA-D1.arff", 1025), "CMATER-BANGLA-D1", 40000, 100));
        streams.add(new ExperimentStream(new ArffFileStream(rootDataDir + "/drift/vis/INTEL-IMGS-D1.arff", 1025), "INTEL-IMGS-D1", 112272, 100));
        streams.add(new ExperimentStream(new ArffFileStream(rootDataDir + "/drift/vis/MNIST-D1.arff", 785), "MNIST-D1", 140000, 100));
        streams.add(new ExperimentStream(new ArffFileStream(rootDataDir + "/drift/vis/IMAGENETTE-D1.arff", 4097), "IMAGENETTE-D1", 37876, 100));
        streams.add(new ExperimentStream(new ArffFileStream(rootDataDir + "/drift/vis/CIFAR10-D1.arff", 1025), "CIFAR10-D1", 120000, 100));
        streams.add(new ExperimentStream(new ArffFileStream(rootDataDir + "/drift/vis/MNIST_F-D1.arff", 785), "MNIST_F-D1", 140000, 100));
        streams.add(new ExperimentStream(new ArffFileStream(rootDataDir + "/drift/vis/MALARIA-D1.arff", 1025), "MALARIA-D1", 55116, 100));
        streams.add(new ExperimentStream(new ArffFileStream(rootDataDir + "/drift/vis/DOGS-VS-CATS-D1.arff", 1025), "DOGS-VS-CATS-D1", 100000, 100));

        return streams;
    }

    private static List<ExperimentStream> createRealDeepStreams(String rootDataDir) {
        List<ExperimentStream> streams = new ArrayList<>();

        // deep-1D, d = 4
        streams.add(new ExperimentStream(new ArffFileStream(rootDataDir + "/text/30k/BBC.arff", 1001), "BBC", 33375, 100));
        streams.add(new ExperimentStream(new ArffFileStream(rootDataDir + "/text/30k/AGNEWS-30K.arff", 170), "AGNEWS-30K", 30000, 100));
        streams.add(new ExperimentStream(new ArffFileStream(rootDataDir + "/generic/30k/SEMG.arff", 3001), "SEMG", 27000, 100));
        streams.add(new ExperimentStream(new ArffFileStream(rootDataDir + "/text/30k/SOGOU-30K.arff", 1501), "SOGOU-30K", 30000, 100));

        // deep-2D, d = 3
        streams.add(new ExperimentStream(new ArffFileStream(rootDataDir + "/vis/32/MNIST.arff", 785), "MNIST", 70000, 100));
        streams.add(new ExperimentStream(new ArffFileStream(rootDataDir + "/vis/32/CMATER-BANGLA.arff", 1025), "CMATER-BANGLA", 20000, 100));
        streams.add(new ExperimentStream(new ArffFileStream(rootDataDir + "/vis/32/INTEL-IMGS.arff", 1025), "INTEL-IMGS", 56136, 100));
        streams.add(new ExperimentStream(new ArffFileStream(rootDataDir + "/vis/64/IMAGENETTE.arff", 4097), "IMAGENETTE", 18938, 100));
        streams.add(new ExperimentStream(new ArffFileStream(rootDataDir + "/vis/32/CIFAR10.arff", 1025), "CIFAR10", 60000, 100));
        streams.add(new ExperimentStream(new ArffFileStream(rootDataDir + "/vis/32/MNIST_F.arff", 785), "MNIST_F", 70000, 100));
        streams.add(new ExperimentStream(new ArffFileStream(rootDataDir + "/vis/32/MALARIA.arff", 1025), "MALARIA", 27558, 100));
        streams.add(new ExperimentStream(new ArffFileStream(rootDataDir + "/vis/32/DOGS-VS-CATS.arff", 1025), "DOGS-VS-CATS", 50000, 100));

        return streams;
    }

}
