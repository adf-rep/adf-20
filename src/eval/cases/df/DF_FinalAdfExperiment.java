package eval.cases.df;

import cls.df.DeepForest;
import cls.df.LayerSize;
import cls.df.base.ARF;
import eval.Evaluator;
import eval.experiment.Experiment;
import eval.experiment.ExperimentRow;
import eval.experiment.ExperimentStream;
import framework.BlindFramework;
import strategies.al.UncertaintyStrategy;
import strategies.al.UncertaintyStrategyType;

import java.util.ArrayList;
import java.util.List;

public class DF_FinalAdfExperiment extends Experiment {

    private int depth = 4; // d = 5 for: 1D streams; d = 2 or 4 for: shallow streams
    private boolean image = true; // false for: 1D and shallow streams
    private boolean append = false; // true for: IMAGENETTE and CIFAR10
    private int numTress = 25; // 40 for: 1D and shallow streams

    public DF_FinalAdfExperiment(String inputDir, String outputDir) {
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

        ARF arf = new ARF();
        arf.ensembleSizeOption.setValue(this.numTress);
        rows.add(new ExperimentRow(
                new BlindFramework(
                        arf,
                        new UncertaintyStrategy(UncertaintyStrategyType.RANDOM, 1.0)
                ),
                "DF", "ARF"
        ));

        rows.add(new ExperimentRow(
                new BlindFramework(
                        new DeepForest(
                                new LayerSize(2, this.numTress, this.depth),
                                false,
                                true,
                                true,
                                true),
                        new UncertaintyStrategy(UncertaintyStrategyType.RANDOM, 1.0)
                ),
                "DF", "CARF"
        ));

        rows.add(new ExperimentRow(
                new BlindFramework(
                        new DeepForest(
                                new LayerSize(2, this.numTress, this.depth),
                                new LayerSize(2, this.numTress, this.depth),
                                -1,
                                false,
                                true,
                                this.append,
                                this.image),
                        new UncertaintyStrategy(UncertaintyStrategyType.RANDOM, 1.0)
                ),
                "DF", "ADF"
        ));

        return rows;
    }

}
