package eval.cases.df;

import cls.df.DeepForest;
import cls.df.LayerSize;
import eval.Evaluator;
import eval.experiment.Experiment;
import eval.experiment.ExperimentRow;
import eval.experiment.ExperimentStream;
import framework.BlindFramework;
import strategies.al.UncertaintyStrategy;
import strategies.al.UncertaintyStrategyType;

import java.util.ArrayList;
import java.util.List;

public class DF_AppendExperiment extends Experiment {

    private int depth = 3; // d = 4 for 1D
    private boolean image = true; // false for 1D

    public DF_AppendExperiment(String inputDir, String outputDir) {
        this.inputDir = inputDir;
        this.outputDir = outputDir;
    }

    @Override
    public void run(Evaluator evaluator) {
        this.conduct(this.createExperimentRows(), ExperimentStream.createExperimentStreams(this.inputDir), evaluator, this.outputDir);
    }

    @Override
    public List<ExperimentRow> createExperimentRows() {
        List<ExperimentRow> rows = new ArrayList<>();

        rows.add(new ExperimentRow(
                new BlindFramework(
                        new DeepForest(
                                new LayerSize(2, 25, this.depth),
                                new LayerSize(2, 25, this.depth),
                                -1,
                                false,
                                true,
                                true,
                                this.image),
                        new UncertaintyStrategy(UncertaintyStrategyType.RANDOM, 1.0)
                ),
                "DF", "ADF-ap"
        ));

        rows.add(new ExperimentRow(
                new BlindFramework(
                        new DeepForest(
                                new LayerSize(2, 25, this.depth),
                                new LayerSize(2, 25, this.depth),
                                -1,
                                false,
                                true,
                                false,
                                this.image),
                        new UncertaintyStrategy(UncertaintyStrategyType.RANDOM, 1.0)
                ),
                "DF", "ADF-nap"
        ));

        return rows;
    }
}
