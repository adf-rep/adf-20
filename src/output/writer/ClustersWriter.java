package output.writer;

import com.yahoo.labs.samoa.instances.Instance;
import javafx.util.Pair;

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;

public class ClustersWriter extends OutputWriter {

    private List<Pair<Instance, Double>> clusters;

    public ClustersWriter(List<Pair<Instance, Double>> clusters) {
        this.clusters = clusters;
    }

    @Override
    public void write(BufferedWriter writer) throws IOException {
        List<String> outClusters = new LinkedList<>();

        for (Pair<Instance, Double> cluster : this.clusters) {
            StringBuilder outCluster = new StringBuilder();

            for (int i = 0; i < cluster.getKey().numAttributes() - 1; i++) {
                outCluster.append(cluster.getKey().value(i)).append(" ");
            }

            outCluster.append(cluster.getValue());
            outClusters.add(outCluster.toString());
        }

        writer.write(String.join(", ", outClusters));
    }

}
