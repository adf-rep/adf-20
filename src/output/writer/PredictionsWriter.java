package output.writer;

import com.yahoo.labs.samoa.instances.Instance;
import framework.FrameworkUpdate;

import java.io.BufferedWriter;
import java.io.IOException;

public class PredictionsWriter extends OutputWriter {

    private Instance prediction;
    private FrameworkUpdate update;

    public PredictionsWriter(Instance prediction, FrameworkUpdate update) {
        this.prediction = prediction;
        this.update = update;
    }

    @Override
    public void write(BufferedWriter writer) throws IOException {
        String predictionStr = prediction.toString();
        writer.write(predictionStr.substring(0, predictionStr.length() - 1) + this.update.toString());
    }

}
