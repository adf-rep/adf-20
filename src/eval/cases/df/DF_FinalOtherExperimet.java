package eval.cases.df;

import eval.Evaluator;
import eval.experiment.Experiment;
import eval.experiment.ExperimentRow;
import eval.experiment.ExperimentStream;
import framework.BlindFramework;
import moa.classifiers.bayes.NaiveBayes;
import moa.classifiers.functions.SGDMultiClass;
import moa.classifiers.meta.*;
import moa.classifiers.trees.HoeffdingAdaptiveTree;
import strategies.al.UncertaintyStrategy;
import strategies.al.UncertaintyStrategyType;

import java.util.ArrayList;
import java.util.List;

public class DF_FinalOtherExperimet extends Experiment {

    public DF_FinalOtherExperimet(String inputDir, String outputDir) {
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
                        new LeveragingBag(),
                        new UncertaintyStrategy(UncertaintyStrategyType.RANDOM, 1.0)
                ),
                "DF", "LB-1.0"
        ));

        rows.add(new ExperimentRow(
                new BlindFramework(
                        new OzaBagAdwin(),
                        new UncertaintyStrategy(UncertaintyStrategyType.RANDOM, 1.0)
                ),
                "DF", "OBAG-ADW-1.0"
        ));

        rows.add(new ExperimentRow(
                new BlindFramework(
                        new OzaBoost(),
                        new UncertaintyStrategy(UncertaintyStrategyType.RANDOM, 1.0)
                ),
                "DF", "OBOS-ADW-1.0"
        ));

        rows.add(new ExperimentRow(
                new BlindFramework(
                        new OnlineSmoothBoost(),
                        new UncertaintyStrategy(UncertaintyStrategyType.RANDOM, 1.0)
                ),
                "DF", "OSB-1.0"
        ));

        rows.add(new ExperimentRow(
                new BlindFramework(
                        new HoeffdingAdaptiveTree(),
                        new UncertaintyStrategy(UncertaintyStrategyType.RANDOM, 1.0)
                ),
                "DF", "HAT-1.0"
        ));

        rows.add(new ExperimentRow(
                new BlindFramework(
                        new SGDMultiClass(),
                        new UncertaintyStrategy(UncertaintyStrategyType.RANDOM, 1.0)
                ),
                "DF", "SGD-1.0"
        ));

        rows.add(new ExperimentRow(
                new BlindFramework(
                        new DynamicWeightedMajority(),
                        new UncertaintyStrategy(UncertaintyStrategyType.RANDOM, 1.0)
                ),
                "DF", "DWM-1.0"
        ));

        rows.add(new ExperimentRow(
                new BlindFramework(
                        new AccuracyUpdatedEnsemble(),
                        new UncertaintyStrategy(UncertaintyStrategyType.RANDOM, 1.0)
                ),
                "DF", "AUC-1.0"
        ));

        rows.add(new ExperimentRow(
                new BlindFramework(
                        new AccuracyWeightedEnsemble(),
                        new UncertaintyStrategy(UncertaintyStrategyType.RANDOM, 1.0)
                ),
                "DF", "AWE-1.0"
        ));

        rows.add(new ExperimentRow(
                new BlindFramework(
                        new LearnNSE(),
                        new UncertaintyStrategy(UncertaintyStrategyType.RANDOM, 1.0)
                ),
                "DF", "LNSE-1.0"
        ));

        rows.add(new ExperimentRow(
                new BlindFramework(
                        new NaiveBayes(),
                        new UncertaintyStrategy(UncertaintyStrategyType.RANDOM, 1.0)
                ),
                "DF", "NB-1.0"
        ));

        return rows;
    }
    
}
