package eval;

import eval.cases.df.*;
import eval.evaluators.BalancedEvaluator;
import eval.experiment.ExperimentResult;
import eval.experiment.ExperimentRow;
import eval.experiment.ExperimentStream;

import java.util.*;

public interface Evaluator {

    static void runAdaptiveDeepForestExperiments(String inputDir, String rootOutputDir) {
        Evaluator evaluator = new BalancedEvaluator();

        // final
        (new DF_FinalAdfExperiment(inputDir, rootOutputDir + "/final-adf")).run(evaluator);
        (new DF_FinalOtherExperimet(inputDir, rootOutputDir + "/final-other")).run(evaluator);

        // settings
//        (new DF_NumTreesExperiment(inputDir, rootOutputDir + "/num-trees")).run(evaluator);
//        (new DF_NumForestsExperiment(inputDir, rootOutputDir + "/num-forests")).run(evaluator);
//        (new DF_AppendExperiment(inputDir, rootOutputDir + "/append")).run(evaluator);
    }

    ExperimentResult evaluate(ExperimentRow experimentRow, ExperimentStream experimentStream);

    static void main(String[] args) {
        String home = System.getProperty("user.home");
        System.out.println("Starting for ADF: " + new Date());

        Evaluator.runAdaptiveDeepForestExperiments(home + "/Data/streams", home + "/Results/adf");

        System.out.println("\nFinished: " + new Date());
        System.exit(0);

    }

}
