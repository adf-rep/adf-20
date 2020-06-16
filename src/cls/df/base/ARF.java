package cls.df.base;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.Instance;
import javafx.util.Pair;
import moa.AbstractMOAObject;
import moa.capabilities.CapabilitiesHandler;
import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.core.DoubleVector;
import moa.core.InstanceExample;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.options.ClassOption;

import java.util.ArrayList;
import java.util.Collection;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/* Disclaimer: This a copy of the original AdaptiveRandomForest classifier implemented in MOA. */

public class ARF extends AbstractClassifier implements MultiClassClassifier, CapabilitiesHandler {
    private static final long serialVersionUID = 1L;
    public ClassOption treeLearnerOption = new ClassOption("treeLearner", 'l', "Random Forest Tree.", ARFHT.class, "ARFHT -e 2000000 -g 50 -c 0.01");
    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's', "The number of trees.", 10, 1, 2147483647);
    public MultiChoiceOption mFeaturesModeOption = new MultiChoiceOption("mFeaturesMode", 'o', "Defines how m, defined by mFeaturesPerTreeSize, is interpreted. M represents the total number of features.", new String[]{"Specified m (integer value)", "sqrt(M)+1", "M-(sqrt(M)+1)", "Percentage (M * (m / 100))"}, new String[]{"SpecifiedM", "SqrtM1", "MSqrtM1", "Percentage"}, 1);
    public IntOption mFeaturesPerTreeSizeOption = new IntOption("mFeaturesPerTreeSize", 'm', "Number of features allowed considered for each split. Negative values corresponds to M - m", 2, -2147483648, 2147483647);
    public FloatOption lambdaOption = new FloatOption("lambda", 'a', "The lambda parameter for bagging.", 6.0D, 1.0D, 3.4028234663852886E38D);
    public IntOption numberOfJobsOption = new IntOption("numberOfJobs", 'j', "Total number of concurrent jobs used for processing (-1 = as much as possible, 0 = do not use multithreading)", 1, -1, 2147483647);
    public ClassOption driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'x', "Change detector for drifts and its parameters", ChangeDetector.class, "ADWINChangeDetector -a 1.0E-5");
    public ClassOption warningDetectionMethodOption = new ClassOption("warningDetectionMethod", 'p', "Change detector for warnings (start training bkg learner)", ChangeDetector.class, "ADWINChangeDetector -a 1.0E-4");
    public FlagOption disableWeightedVote = new FlagOption("disableWeightedVote", 'w', "Should use weighted voting?");
    public FlagOption disableDriftDetectionOption = new FlagOption("disableDriftDetection", 'u', "Should use drift detection? If disabled then bkg learner is also disabled");
    public FlagOption disableBackgroundLearnerOption = new FlagOption("disableBackgroundLearner", 'q', "Should use bkg learner? If disabled then reset tree immediately.");
    protected static final int FEATURES_M = 0;
    protected static final int FEATURES_SQRT = 1;
    protected static final int FEATURES_SQRT_INV = 2;
    protected static final int FEATURES_PERCENT = 3;
    protected static final int SINGLE_THREAD = 0;
    protected ARF.ARFBaseLearner[] ensemble;
    protected long instancesSeen;
    protected int subspaceSize;
    protected BasicClassificationPerformanceEvaluator evaluator;
    private ExecutorService executor;

    public ARF() {
    }

    public String getPurposeString() {
        return "Adaptive Random Forest algorithm for evolving data streams from Gomes et al.";
    }

    public void resetLearningImpl() {
        this.ensemble = null;
        this.subspaceSize = 0;
        this.instancesSeen = 0L;
        this.evaluator = new BasicClassificationPerformanceEvaluator();
        int numberOfJobs;
        if (this.numberOfJobsOption.getValue() == -1) {
            numberOfJobs = Runtime.getRuntime().availableProcessors();
        } else {
            numberOfJobs = this.numberOfJobsOption.getValue();
        }

        if (numberOfJobs != 0 && numberOfJobs != 1) {
            this.executor = Executors.newFixedThreadPool(numberOfJobs);
        }

    }

    public void trainOnInstanceImpl(Instance instance) {
        ++this.instancesSeen;
        if (this.ensemble == null) {
            this.initEnsemble(instance);
        }

        Collection<ARF.TrainingRunnable> trainers = new ArrayList();

        for(int i = 0; i < this.ensemble.length; ++i) {
            DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(instance));
            InstanceExample example = new InstanceExample(instance);
            this.ensemble[i].evaluator.addResult(example, vote.getArrayRef());
            int k = MiscUtils.poisson(this.lambdaOption.getValue(), this.classifierRandom);
            if (k > 0) {
                if (this.executor != null) {
                    ARF.TrainingRunnable trainer = new ARF.TrainingRunnable(this.ensemble[i], instance, (double)k, this.instancesSeen);
                    trainers.add(trainer);
                } else {
                    this.ensemble[i].trainOnInstance(instance, (double)k, this.instancesSeen);
                }
            }
        }

        if (this.executor != null) {
            try {
                this.executor.invokeAll(trainers);
            } catch (InterruptedException var8) {
                throw new RuntimeException("Could not call invokeAll() on training threads.");
            }
        }

    }

    public double[] getVotesForInstance(Instance instance) {
        Instance testInstance = instance.copy();
        if (this.ensemble == null) {
            this.initEnsemble(testInstance);
        }

        DoubleVector combinedVote = new DoubleVector();

        for(int i = 0; i < this.ensemble.length; ++i) {
            DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(testInstance));
            if (vote.sumOfValues() > 0.0D) {
                vote.normalize();
                double acc = this.ensemble[i].evaluator.getPerformanceMeasurements()[1].getValue();
                if (!this.disableWeightedVote.isSet() && acc > 0.0D) {
                    for(int v = 0; v < vote.numValues(); ++v) {
                        vote.setValue(v, vote.getValue(v) * acc);
                    }
                }

                combinedVote.addValues(vote);
            }
        }

        return combinedVote.getArrayRef();
    }

    public boolean isRandomizable() {
        return true;
    }

    public void getModelDescription(StringBuilder arg0, int arg1) {
    }

    public Pair<Double, Integer> getDepthStats() {
        int maxDepth = 0;
        double avg = 0.0;

        if (this.ensemble != null) {
            for (int i = 0; i < this.ensemble.length; i++) {
                int depth = this.ensemble[i].classifier.measureTreeDepth();
                avg += depth;

                if (depth > maxDepth) {
                    maxDepth = depth;
                }
            }

            avg = avg / this.ensemble.length;
        }

        return new Pair<>(avg, maxDepth);
    }

    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    protected void initEnsemble(Instance instance) {
        int ensembleSize = this.ensembleSizeOption.getValue();
        this.ensemble = new ARF.ARFBaseLearner[ensembleSize];
        BasicClassificationPerformanceEvaluator classificationEvaluator = new BasicClassificationPerformanceEvaluator();
        this.subspaceSize = this.mFeaturesPerTreeSizeOption.getValue();
        int n = instance.numAttributes() - 1;
        switch(this.mFeaturesModeOption.getChosenIndex()) {
            case 1:
                this.subspaceSize = (int)Math.round(Math.sqrt((double)n)) + 1;
                break;
            case 2:
                this.subspaceSize = n - (int)Math.round(Math.sqrt((double)n) + 1.0D);
                break;
            case 3:
                double percent = this.subspaceSize < 0 ? (double)(100 + this.subspaceSize) / 100.0D : (double)this.subspaceSize / 100.0D;
                this.subspaceSize = (int)Math.round((double)n * percent);
        }

        if (this.subspaceSize < 0) {
            this.subspaceSize += n;
        }

        if (this.subspaceSize <= 0) {
            this.subspaceSize = 1;
        }

        if (this.subspaceSize > n) {
            this.subspaceSize = n;
        }

        ARFHT treeLearner = (ARFHT)this.getPreparedClassOption(this.treeLearnerOption);
        treeLearner.resetLearning();

        for(int i = 0; i < ensembleSize; ++i) {
            treeLearner.subspaceSizeOption.setValue(this.subspaceSize);
            this.ensemble[i] = new ARF.ARFBaseLearner(i, (ARFHT)treeLearner.copy(), (BasicClassificationPerformanceEvaluator)classificationEvaluator.copy(), this.instancesSeen, !this.disableBackgroundLearnerOption.isSet(), !this.disableDriftDetectionOption.isSet(), this.driftDetectionMethodOption, this.warningDetectionMethodOption, false);
        }

    }

    public ImmutableCapabilities defineImmutableCapabilities() {
        return this.getClass() == ARF.class ? new ImmutableCapabilities(new Capability[]{Capability.VIEW_STANDARD, Capability.VIEW_LITE}) : new ImmutableCapabilities(new Capability[]{Capability.VIEW_STANDARD});
    }

    protected class TrainingRunnable implements Runnable, Callable<Integer> {
        private final ARF.ARFBaseLearner learner;
        private final Instance instance;
        private final double weight;
        private final long instancesSeen;

        public TrainingRunnable(ARF.ARFBaseLearner learner, Instance instance, double weight, long instancesSeen) {
            this.learner = learner;
            this.instance = instance;
            this.weight = weight;
            this.instancesSeen = instancesSeen;
        }

        public void run() {
            this.learner.trainOnInstance(this.instance, this.weight, this.instancesSeen);
        }

        public Integer call() throws Exception {
            this.run();
            return 0;
        }
    }

    protected final class ARFBaseLearner extends AbstractMOAObject {
        public int indexOriginal;
        public long createdOn;
        public long lastDriftOn;
        public long lastWarningOn;
        public ARFHT classifier;
        public boolean isBackgroundLearner;
        protected ClassOption driftOption;
        protected ClassOption warningOption;
        protected ChangeDetector driftDetectionMethod;
        protected ChangeDetector warningDetectionMethod;
        public boolean useBkgLearner;
        public boolean useDriftDetector;
        protected ARF.ARFBaseLearner bkgLearner;
        public BasicClassificationPerformanceEvaluator evaluator;
        protected int numberOfDriftsDetected;
        protected int numberOfWarningsDetected;

        private void init(int indexOriginal, ARFHT instantiatedClassifier, BasicClassificationPerformanceEvaluator evaluatorInstantiated, long instancesSeen, boolean useBkgLearner, boolean useDriftDetector, ClassOption driftOption, ClassOption warningOption, boolean isBackgroundLearner) {
            this.indexOriginal = indexOriginal;
            this.createdOn = instancesSeen;
            this.lastDriftOn = 0L;
            this.lastWarningOn = 0L;
            this.classifier = instantiatedClassifier;
            this.evaluator = evaluatorInstantiated;
            this.useBkgLearner = useBkgLearner;
            this.useDriftDetector = useDriftDetector;
            this.numberOfDriftsDetected = 0;
            this.numberOfWarningsDetected = 0;
            this.isBackgroundLearner = isBackgroundLearner;
            if (this.useDriftDetector) {
                this.driftOption = driftOption;
                this.driftDetectionMethod = ((ChangeDetector) ARF.this.getPreparedClassOption(this.driftOption)).copy();
            }

            if (this.useBkgLearner) {
                this.warningOption = warningOption;
                this.warningDetectionMethod = ((ChangeDetector) ARF.this.getPreparedClassOption(this.warningOption)).copy();
            }

        }

        public ARFBaseLearner(int indexOriginal, ARFHT instantiatedClassifier, BasicClassificationPerformanceEvaluator evaluatorInstantiated, long instancesSeen, boolean useBkgLearner, boolean useDriftDetector, ClassOption driftOption, ClassOption warningOption, boolean isBackgroundLearner) {
            this.init(indexOriginal, instantiatedClassifier, evaluatorInstantiated, instancesSeen, useBkgLearner, useDriftDetector, driftOption, warningOption, isBackgroundLearner);
        }

        public void reset() {
            if (this.useBkgLearner && this.bkgLearner != null) {
                this.classifier = this.bkgLearner.classifier;
                this.driftDetectionMethod = this.bkgLearner.driftDetectionMethod;
                this.warningDetectionMethod = this.bkgLearner.warningDetectionMethod;
                this.evaluator = this.bkgLearner.evaluator;
                this.createdOn = this.bkgLearner.createdOn;
                this.bkgLearner = null;
            } else {
                this.classifier.resetLearning();
                this.createdOn = ARF.this.instancesSeen;
                this.driftDetectionMethod = ((ChangeDetector) ARF.this.getPreparedClassOption(this.driftOption)).copy();
            }

            this.evaluator.reset();
        }

        public void trainOnInstance(Instance instance, double weight, long instancesSeen) {
            Instance weightedInstance = instance.copy();
            weightedInstance.setWeight(instance.weight() * weight);
            this.classifier.trainOnInstance(weightedInstance);
            if (this.bkgLearner != null) {
                this.bkgLearner.classifier.trainOnInstance(instance);
            }

            if (this.useDriftDetector && !this.isBackgroundLearner) {
                boolean correctlyClassifies = this.classifier.correctlyClassifies(instance);
                if (this.useBkgLearner) {
                    this.warningDetectionMethod.input(correctlyClassifies ? 0.0D : 1.0D);
                    if (this.warningDetectionMethod.getChange()) {
                        this.lastWarningOn = instancesSeen;
                        ++this.numberOfWarningsDetected;
                        ARFHT bkgClassifier = (ARFHT)this.classifier.copy();
                        bkgClassifier.resetLearning();
                        BasicClassificationPerformanceEvaluator bkgEvaluator = (BasicClassificationPerformanceEvaluator)this.evaluator.copy();
                        bkgEvaluator.reset();
                        this.bkgLearner = ARF.this.new ARFBaseLearner(this.indexOriginal, bkgClassifier, bkgEvaluator, instancesSeen, this.useBkgLearner, this.useDriftDetector, this.driftOption, this.warningOption, true);
                        this.warningDetectionMethod = ((ChangeDetector) ARF.this.getPreparedClassOption(this.warningOption)).copy();
                    }
                }

                this.driftDetectionMethod.input(correctlyClassifies ? 0.0D : 1.0D);
                if (this.driftDetectionMethod.getChange()) {
                    this.lastDriftOn = instancesSeen;
                    ++this.numberOfDriftsDetected;
                    this.reset();
                }
            }

        }

        public double[] getVotesForInstance(Instance instance) {
            DoubleVector vote = new DoubleVector(this.classifier.getVotesForInstance(instance));
            return vote.getArrayRef();
        }

        public void getDescription(StringBuilder sb, int indent) {
        }
    }
}