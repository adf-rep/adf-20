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

public class DF_NumTreesExperiment extends Experiment {

    private int depth = 3; // d = 4 for 1D
    private boolean image = true; // false for 1D

    public DF_NumTreesExperiment(String inputDir, String outputDir) {
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
                                new LayerSize(2, 10, this.depth),
                                new LayerSize(2, 10, this.depth),
                                -1,
                                false,
                                true,
                                true,
                                this.image),
                        new UncertaintyStrategy(UncertaintyStrategyType.RANDOM, 1.0)
                ),
                "DF", "ADF-10"
        ));

        rows.add(new ExperimentRow(
                new BlindFramework(
                        new DeepForest(
                                new LayerSize(2, 20, this.depth),
                                new LayerSize(2, 20, this.depth),
                                -1,
                                false,
                                true,
                                true,
                                this.image),
                        new UncertaintyStrategy(UncertaintyStrategyType.RANDOM, 1.0)
                ),
                "DF", "ADF-20"
        ));

        rows.add(new ExperimentRow(
                new BlindFramework(
                        new DeepForest(
                                new LayerSize(2, 30, this.depth),
                                new LayerSize(2, 30, this.depth),
                                -1,
                                false,
                                true,
                                true,
                                this.image),
                        new UncertaintyStrategy(UncertaintyStrategyType.RANDOM, 1.0)
                ),
                "DF", "ADF-30"
        ));

        rows.add(new ExperimentRow(
                new BlindFramework(
                        new DeepForest(
                                new LayerSize(2, 40, this.depth),
                                new LayerSize(2, 40, this.depth),
                                -1,
                                false,
                                true,
                                true,
                                this.image),
                        new UncertaintyStrategy(UncertaintyStrategyType.RANDOM, 1.0)
                ),
                "DF", "ADF-40"
        ));

        rows.add(new ExperimentRow(
                new BlindFramework(
                        new DeepForest(
                                new LayerSize(2, 50, this.depth),
                                new LayerSize(2, 50, this.depth),
                                -1,
                                false,
                                true,
                                true,
                                this.image),
                        new UncertaintyStrategy(UncertaintyStrategyType.RANDOM, 1.0)
                ),
                "DF", "ADF-50"
        ));

        return rows;
    }

}
