package cls.df.base;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.AbstractMOAObject;
import moa.capabilities.CapabilitiesHandler;
import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.bayes.NaiveBayes;
import moa.classifiers.core.AttributeSplitSuggestion;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.DiscreteAttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.NullAttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.NumericAttributeClassObserver;
import moa.classifiers.core.conditionaltests.InstanceConditionalTest;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.core.*;
import moa.options.ClassOption;

import java.util.*;

/* Disclaimer: This a copy of the original HoeffdingTree classifier implemented in MOA. */

public class HT extends AbstractClassifier implements MultiClassClassifier, CapabilitiesHandler {
    private static final long serialVersionUID = 1L;
    public IntOption maxByteSizeOption = new IntOption("maxByteSize", 'm', "Maximum memory consumed by the tree.", 33554432, 0, 2147483647);
    public ClassOption numericEstimatorOption = new ClassOption("numericEstimator", 'n', "Numeric estimator to use.", NumericAttributeClassObserver.class, "GaussianNumericAttributeClassObserver");
    public ClassOption nominalEstimatorOption = new ClassOption("nominalEstimator", 'd', "Nominal estimator to use.", DiscreteAttributeClassObserver.class, "NominalAttributeClassObserver");
    public IntOption memoryEstimatePeriodOption = new IntOption("memoryEstimatePeriod", 'e', "How many instances between memory consumption checks.", 1000000, 0, 2147483647);
    public IntOption gracePeriodOption = new IntOption("gracePeriod", 'g', "The number of instances a leaf should observe between split attempts.", 200, 0, 2147483647);
    public ClassOption splitCriterionOption = new ClassOption("splitCriterion", 's', "Split criterion to use.", SplitCriterion.class, "InfoGainSplitCriterion");
    public FloatOption splitConfidenceOption = new FloatOption("splitConfidence", 'c', "The allowable error in split decision, values closer to 0 will take longer to decide.", 1.0E-7D, 0.0D, 1.0D);
    public FloatOption tieThresholdOption = new FloatOption("tieThreshold", 't', "Threshold below which a split will be forced to break ties.", 0.05D, 0.0D, 1.0D);
    public FlagOption binarySplitsOption = new FlagOption("binarySplits", 'b', "Only allow binary splits.");
    public FlagOption stopMemManagementOption = new FlagOption("stopMemManagement", 'z', "Stop growing as soon as memory limit is hit.");
    public FlagOption removePoorAttsOption = new FlagOption("removePoorAtts", 'r', "Disable poor attributes.");
    public FlagOption noPrePruneOption = new FlagOption("noPrePrune", 'p', "Disable pre-pruning.");
    protected HT.Node treeRoot;
    protected int decisionNodeCount;
    protected int activeLeafNodeCount;
    protected int inactiveLeafNodeCount;
    protected double inactiveLeafByteSizeEstimate;
    protected double activeLeafByteSizeEstimate;
    protected double byteSizeEstimateOverheadFraction;
    protected boolean growthAllowed;
    public MultiChoiceOption leafpredictionOption = new MultiChoiceOption("leafprediction", 'l', "Leaf prediction to use.", new String[]{"MC", "NB", "NBAdaptive"}, new String[]{"Majority class", "Naive Bayes", "Naive Bayes Adaptive"}, 2);
    public IntOption nbThresholdOption = new IntOption("nbThreshold", 'q', "The number of instances a leaf should observe before permitting Naive Bayes.", 0, 0, 2147483647);

    public HT() {
    }

    public String getPurposeString() {
        return "Hoeffding Tree or VFDT.";
    }

    public int calcByteSize() {
        int size = (int) SizeOf.sizeOf(this);
        if (this.treeRoot != null) {
            size += this.treeRoot.calcByteSizeIncludingSubtree();
        }

        return size;
    }

    public int measureByteSize() {
        return this.calcByteSize();
    }

    public void resetLearningImpl() {
        this.treeRoot = null;
        this.decisionNodeCount = 0;
        this.activeLeafNodeCount = 0;
        this.inactiveLeafNodeCount = 0;
        this.inactiveLeafByteSizeEstimate = 0.0D;
        this.activeLeafByteSizeEstimate = 0.0D;
        this.byteSizeEstimateOverheadFraction = 1.0D;
        this.growthAllowed = true;
        if (this.leafpredictionOption.getChosenIndex() > 0) {
            this.removePoorAttsOption = null;
        }

    }

    public void trainOnInstanceImpl(Instance inst) {
        if (this.treeRoot == null) {
            this.treeRoot = this.newLearningNode();
            this.activeLeafNodeCount = 1;
        }

        HT.FoundNode foundNode = this.treeRoot.filterInstanceToLeaf(inst, (HT.SplitNode)null, -1);
        HT.Node leafNode = foundNode.node;
        if (leafNode == null) {
            leafNode = this.newLearningNode();
            foundNode.parent.setChild(foundNode.parentBranch, (HT.Node)leafNode);
            ++this.activeLeafNodeCount;
        }

        if (leafNode instanceof LearningNode) {
            LearningNode learningNode = (LearningNode)leafNode;
            learningNode.learnFromInstance(inst, this);
            if (this.growthAllowed && learningNode instanceof ActiveLearningNode) {
                ActiveLearningNode activeLearningNode = (ActiveLearningNode)learningNode;
                double weightSeen = activeLearningNode.getWeightSeen();
                if (weightSeen - activeLearningNode.getWeightSeenAtLastSplitEvaluation() >= (double)this.gracePeriodOption.getValue()) {
                    this.attemptToSplit(activeLearningNode, foundNode.parent, foundNode.parentBranch);
                    activeLearningNode.setWeightSeenAtLastSplitEvaluation(weightSeen);
                }
            }
        }

        if (this.trainingWeightSeenByModel % (double)this.memoryEstimatePeriodOption.getValue() == 0.0D) {
            this.estimateModelByteSizes();
        }

    }

    public double[] getVotesForInstance(Instance inst) {
        if (this.treeRoot != null) {
            HT.FoundNode foundNode = this.treeRoot.filterInstanceToLeaf(inst, (HT.SplitNode)null, -1);
            HT.Node leafNode = foundNode.node;
            if (leafNode == null) {
                leafNode = foundNode.parent;
            }

            return ((HT.Node)leafNode).getClassVotes(inst, this);
        } else {
            int numClasses = inst.dataset().numClasses();
            return new double[numClasses];
        }
    }

    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[]{new Measurement("tree size (nodes)", (double)(this.decisionNodeCount + this.activeLeafNodeCount + this.inactiveLeafNodeCount)), new Measurement("tree size (leaves)", (double)(this.activeLeafNodeCount + this.inactiveLeafNodeCount)), new Measurement("active learning leaves", (double)this.activeLeafNodeCount), new Measurement("tree depth", (double)this.measureTreeDepth()), new Measurement("active leaf byte size estimate", this.activeLeafByteSizeEstimate), new Measurement("inactive leaf byte size estimate", this.inactiveLeafByteSizeEstimate), new Measurement("byte size estimate overhead", this.byteSizeEstimateOverheadFraction)};
    }

    protected int measureTreeDepth() {
        return this.treeRoot != null ? this.treeRoot.subtreeDepth() : 0;
    }

    public void getModelDescription(StringBuilder out, int indent) {
        this.treeRoot.describeSubtree(this, out, indent);
    }

    public boolean isRandomizable() {
        return false;
    }

    public static double computeHoeffdingBound(double range, double confidence, double n) {
        return Math.sqrt(range * range * Math.log(1.0D / confidence) / (2.0D * n));
    }

    protected HT.SplitNode newSplitNode(InstanceConditionalTest splitTest, double[] classObservations, int size) {
        return new HT.SplitNode(splitTest, classObservations, size);
    }

    protected HT.SplitNode newSplitNode(InstanceConditionalTest splitTest, double[] classObservations) {
        return new HT.SplitNode(splitTest, classObservations);
    }

    protected AttributeClassObserver newNominalClassObserver() {
        AttributeClassObserver nominalClassObserver = (AttributeClassObserver)this.getPreparedClassOption(this.nominalEstimatorOption);
        return (AttributeClassObserver)nominalClassObserver.copy();
    }

    protected AttributeClassObserver newNumericClassObserver() {
        AttributeClassObserver numericClassObserver = (AttributeClassObserver)this.getPreparedClassOption(this.numericEstimatorOption);
        return (AttributeClassObserver)numericClassObserver.copy();
    }

    protected void attemptToSplit(ActiveLearningNode node, HT.SplitNode parent, int parentIndex) {
        if (!node.observedClassDistributionIsPure()) {
            SplitCriterion splitCriterion = (SplitCriterion)this.getPreparedClassOption(this.splitCriterionOption);
            AttributeSplitSuggestion[] bestSplitSuggestions = node.getBestSplitSuggestions(splitCriterion, this);
            Arrays.sort(bestSplitSuggestions);
            boolean shouldSplit = false;
            if (bestSplitSuggestions.length < 2) {
                shouldSplit = bestSplitSuggestions.length > 0;
            } else {
                double hoeffdingBound = computeHoeffdingBound(splitCriterion.getRangeOfMerit(node.getObservedClassDistribution()), this.splitConfidenceOption.getValue(), node.getWeightSeen());
                AttributeSplitSuggestion bestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 1];
                AttributeSplitSuggestion secondBestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 2];
                if (bestSuggestion.merit - secondBestSuggestion.merit > hoeffdingBound || hoeffdingBound < this.tieThresholdOption.getValue()) {
                    shouldSplit = true;
                }

                if (this.removePoorAttsOption != null && this.removePoorAttsOption.isSet()) {
                    Set<Integer> poorAtts = new HashSet();

                    int i;
                    int[] splitAtts;
                    for(i = 0; i < bestSplitSuggestions.length; ++i) {
                        if (bestSplitSuggestions[i].splitTest != null) {
                            splitAtts = bestSplitSuggestions[i].splitTest.getAttsTestDependsOn();
                            if (splitAtts.length == 1 && bestSuggestion.merit - bestSplitSuggestions[i].merit > hoeffdingBound) {
                                poorAtts.add(new Integer(splitAtts[0]));
                            }
                        }
                    }

                    for(i = 0; i < bestSplitSuggestions.length; ++i) {
                        if (bestSplitSuggestions[i].splitTest != null) {
                            splitAtts = bestSplitSuggestions[i].splitTest.getAttsTestDependsOn();
                            if (splitAtts.length == 1 && bestSuggestion.merit - bestSplitSuggestions[i].merit < hoeffdingBound) {
                                poorAtts.remove(new Integer(splitAtts[0]));
                            }
                        }
                    }

                    Iterator var17 = poorAtts.iterator();

                    while(var17.hasNext()) {
                        int poorAtt = (Integer)var17.next();
                        node.disableAttribute(poorAtt);
                    }
                }
            }

            if (shouldSplit) {
                AttributeSplitSuggestion splitDecision = bestSplitSuggestions[bestSplitSuggestions.length - 1];
                if (splitDecision.splitTest == null) {
                    this.deactivateLearningNode(node, parent, parentIndex);
                } else {
                    HT.SplitNode newSplit = this.newSplitNode(splitDecision.splitTest, node.getObservedClassDistribution(), splitDecision.numSplits());

                    for(int i = 0; i < splitDecision.numSplits(); ++i) {
                        HT.Node newChild = this.newLearningNode(splitDecision.resultingClassDistributionFromSplit(i));
                        newSplit.setChild(i, newChild);
                    }

                    --this.activeLeafNodeCount;
                    ++this.decisionNodeCount;
                    this.activeLeafNodeCount += splitDecision.numSplits();
                    if (parent == null) {
                        this.treeRoot = newSplit;
                    } else {
                        parent.setChild(parentIndex, newSplit);
                    }
                }

                this.enforceTrackerLimit();
            }
        }

    }

    public void enforceTrackerLimit() {
        if (this.inactiveLeafNodeCount > 0 || ((double)this.activeLeafNodeCount * this.activeLeafByteSizeEstimate + (double)this.inactiveLeafNodeCount * this.inactiveLeafByteSizeEstimate) * this.byteSizeEstimateOverheadFraction > (double)this.maxByteSizeOption.getValue()) {
            if (this.stopMemManagementOption.isSet()) {
                this.growthAllowed = false;
                return;
            }

            HT.FoundNode[] learningNodes = this.findLearningNodes();
            Arrays.sort(learningNodes, new Comparator<HT.FoundNode>() {
                public int compare(HT.FoundNode fn1, HT.FoundNode fn2) {
                    return Double.compare(fn1.node.calculatePromise(), fn2.node.calculatePromise());
                }
            });
            int maxActive = 0;

            while(maxActive < learningNodes.length) {
                ++maxActive;
                if (((double)maxActive * this.activeLeafByteSizeEstimate + (double)(learningNodes.length - maxActive) * this.inactiveLeafByteSizeEstimate) * this.byteSizeEstimateOverheadFraction > (double)this.maxByteSizeOption.getValue()) {
                    --maxActive;
                    break;
                }
            }

            int cutoff = learningNodes.length - maxActive;

            int i;
            for(i = 0; i < cutoff; ++i) {
                if (learningNodes[i].node instanceof ActiveLearningNode) {
                    this.deactivateLearningNode((ActiveLearningNode)learningNodes[i].node, learningNodes[i].parent, learningNodes[i].parentBranch);
                }
            }

            for(i = cutoff; i < learningNodes.length; ++i) {
                if (learningNodes[i].node instanceof HT.InactiveLearningNode) {
                    this.activateLearningNode((HT.InactiveLearningNode)learningNodes[i].node, learningNodes[i].parent, learningNodes[i].parentBranch);
                }
            }
        }

    }

    public void estimateModelByteSizes() {
        HT.FoundNode[] learningNodes = this.findLearningNodes();
        long totalActiveSize = 0L;
        long totalInactiveSize = 0L;
        HT.FoundNode[] var6 = learningNodes;
        int var7 = learningNodes.length;

        for(int var8 = 0; var8 < var7; ++var8) {
            HT.FoundNode foundNode = var6[var8];
            if (foundNode.node instanceof ActiveLearningNode) {
                totalActiveSize += SizeOf.fullSizeOf(foundNode.node);
            } else {
                totalInactiveSize += SizeOf.fullSizeOf(foundNode.node);
            }
        }

        if (totalActiveSize > 0L) {
            this.activeLeafByteSizeEstimate = (double)totalActiveSize / (double)this.activeLeafNodeCount;
        }

        if (totalInactiveSize > 0L) {
            this.inactiveLeafByteSizeEstimate = (double)totalInactiveSize / (double)this.inactiveLeafNodeCount;
        }

        int actualModelSize = this.measureByteSize();
        double estimatedModelSize = (double)this.activeLeafNodeCount * this.activeLeafByteSizeEstimate + (double)this.inactiveLeafNodeCount * this.inactiveLeafByteSizeEstimate;
        this.byteSizeEstimateOverheadFraction = (double)actualModelSize / estimatedModelSize;
        if (actualModelSize > this.maxByteSizeOption.getValue()) {
            this.enforceTrackerLimit();
        }

    }

    public void deactivateAllLeaves() {
        HT.FoundNode[] learningNodes = this.findLearningNodes();

        for(int i = 0; i < learningNodes.length; ++i) {
            if (learningNodes[i].node instanceof ActiveLearningNode) {
                this.deactivateLearningNode((ActiveLearningNode)learningNodes[i].node, learningNodes[i].parent, learningNodes[i].parentBranch);
            }
        }

    }

    protected void deactivateLearningNode(ActiveLearningNode toDeactivate, HT.SplitNode parent, int parentBranch) {
        HT.Node newLeaf = new HT.InactiveLearningNode(toDeactivate.getObservedClassDistribution());
        if (parent == null) {
            this.treeRoot = newLeaf;
        } else {
            parent.setChild(parentBranch, newLeaf);
        }

        --this.activeLeafNodeCount;
        ++this.inactiveLeafNodeCount;
    }

    protected void activateLearningNode(HT.InactiveLearningNode toActivate, HT.SplitNode parent, int parentBranch) {
        HT.Node newLeaf = this.newLearningNode(toActivate.getObservedClassDistribution());
        if (parent == null) {
            this.treeRoot = newLeaf;
        } else {
            parent.setChild(parentBranch, newLeaf);
        }

        ++this.activeLeafNodeCount;
        --this.inactiveLeafNodeCount;
    }

    protected HT.FoundNode[] findLearningNodes() {
        List<HT.FoundNode> foundList = new LinkedList();
        this.findLearningNodes(this.treeRoot, (HT.SplitNode)null, -1, foundList);
        return (HT.FoundNode[])foundList.toArray(new HT.FoundNode[foundList.size()]);
    }

    protected void findLearningNodes(HT.Node node, HT.SplitNode parent, int parentBranch, List<HT.FoundNode> found) {
        if (node != null) {
            if (node instanceof LearningNode) {
                found.add(new HT.FoundNode(node, parent, parentBranch));
            }

            if (node instanceof HT.SplitNode) {
                HT.SplitNode splitNode = (HT.SplitNode)node;

                for(int i = 0; i < splitNode.numChildren(); ++i) {
                    this.findLearningNodes(splitNode.getChild(i), splitNode, i, found);
                }
            }
        }

    }

    protected LearningNode newLearningNode() {
        return this.newLearningNode(new double[0]);
    }

    protected LearningNode newLearningNode(double[] initialClassObservations) {
        int predictionOption = this.leafpredictionOption.getChosenIndex();
        Object ret;
        if (predictionOption == 0) {
            ret = new ActiveLearningNode(initialClassObservations);
        } else if (predictionOption == 1) {
            ret = new LearningNodeNB(initialClassObservations);
        } else {
            ret = new LearningNodeNBAdaptive(initialClassObservations);
        }

        return (LearningNode)ret;
    }

    public ImmutableCapabilities defineImmutableCapabilities() {
        return this.getClass() == HT.class ? new ImmutableCapabilities(new Capability[]{Capability.VIEW_STANDARD, Capability.VIEW_LITE}) : new ImmutableCapabilities(new Capability[]{Capability.VIEW_STANDARD});
    }

    public static class LearningNodeNBAdaptive extends LearningNodeNB {
        private static final long serialVersionUID = 1L;
        protected double mcCorrectWeight = 0.0D;
        protected double nbCorrectWeight = 0.0D;

        public LearningNodeNBAdaptive(double[] initialClassObservations) {
            super(initialClassObservations);
        }

        public void learnFromInstance(Instance inst, HT ht) {
            int trueClass = (int)inst.classValue();
            if (this.observedClassDistribution.maxIndex() == trueClass) {
                this.mcCorrectWeight += inst.weight();
            }

            if (Utils.maxIndex(NaiveBayes.doNaiveBayesPrediction(inst, this.observedClassDistribution, this.attributeObservers)) == trueClass) {
                this.nbCorrectWeight += inst.weight();
            }

            super.learnFromInstance(inst, ht);
        }

        public double[] getClassVotes(Instance inst, HT ht) {
            return this.mcCorrectWeight > this.nbCorrectWeight ? this.observedClassDistribution.getArrayCopy() : NaiveBayes.doNaiveBayesPrediction(inst, this.observedClassDistribution, this.attributeObservers);
        }
    }

    public static class LearningNodeNB extends ActiveLearningNode {
        private static final long serialVersionUID = 1L;

        public LearningNodeNB(double[] initialClassObservations) {
            super(initialClassObservations);
        }

        public double[] getClassVotes(Instance inst, HT ht) {
            return this.getWeightSeen() >= (double)ht.nbThresholdOption.getValue() ? NaiveBayes.doNaiveBayesPrediction(inst, this.observedClassDistribution, this.attributeObservers) : super.getClassVotes(inst, ht);
        }

        public void disableAttribute(int attIndex) {
        }
    }

    public static class ActiveLearningNode extends LearningNode {
        private static final long serialVersionUID = 1L;
        protected double weightSeenAtLastSplitEvaluation = this.getWeightSeen();
        protected AutoExpandVector<AttributeClassObserver> attributeObservers = new AutoExpandVector();
        protected boolean isInitialized = false;

        public ActiveLearningNode(double[] initialClassObservations) {
            super(initialClassObservations);
        }

        public int calcByteSize() {
            return super.calcByteSize() + (int)SizeOf.fullSizeOf(this.attributeObservers);
        }

        public void learnFromInstance(Instance inst, HT ht) {
            if (!this.isInitialized) {
                this.attributeObservers = new AutoExpandVector(inst.numAttributes());
                this.isInitialized = true;
            }

            this.observedClassDistribution.addToValue((int)inst.classValue(), inst.weight());

            for(int i = 0; i < inst.numAttributes() - 1; ++i) {
                int instAttIndex = HT.modelAttIndexToInstanceAttIndex(i, inst);
                AttributeClassObserver obs = (AttributeClassObserver)this.attributeObservers.get(i);
                if (obs == null) {
                    obs = inst.attribute(instAttIndex).isNominal() ? ht.newNominalClassObserver() : ht.newNumericClassObserver();
                    this.attributeObservers.set(i, obs);
                }

                obs.observeAttributeClass(inst.value(instAttIndex), (int)inst.classValue(), inst.weight());
            }

        }

        public double getWeightSeen() {
            return this.observedClassDistribution.sumOfValues();
        }

        public double getWeightSeenAtLastSplitEvaluation() {
            return this.weightSeenAtLastSplitEvaluation;
        }

        public void setWeightSeenAtLastSplitEvaluation(double weight) {
            this.weightSeenAtLastSplitEvaluation = weight;
        }

        public AttributeSplitSuggestion[] getBestSplitSuggestions(SplitCriterion criterion, HT ht) {
            List<AttributeSplitSuggestion> bestSuggestions = new LinkedList();
            double[] preSplitDist = this.observedClassDistribution.getArrayCopy();
            if (!ht.noPrePruneOption.isSet()) {
                bestSuggestions.add(new AttributeSplitSuggestion((InstanceConditionalTest)null, new double[0][], criterion.getMeritOfSplit(preSplitDist, new double[][]{preSplitDist})));
            }

            for(int i = 0; i < this.attributeObservers.size(); ++i) {
                AttributeClassObserver obs = (AttributeClassObserver)this.attributeObservers.get(i);
                if (obs != null) {
                    AttributeSplitSuggestion bestSuggestion = obs.getBestEvaluatedSplitSuggestion(criterion, preSplitDist, i, ht.binarySplitsOption.isSet());
                    if (bestSuggestion != null) {
                        bestSuggestions.add(bestSuggestion);
                    }
                }
            }

            return (AttributeSplitSuggestion[])bestSuggestions.toArray(new AttributeSplitSuggestion[bestSuggestions.size()]);
        }

        public void disableAttribute(int attIndex) {
            this.attributeObservers.set(attIndex, new NullAttributeClassObserver());
        }
    }

    public static class InactiveLearningNode extends LearningNode {
        private static final long serialVersionUID = 1L;

        public InactiveLearningNode(double[] initialClassObservations) {
            super(initialClassObservations);
        }

        public void learnFromInstance(Instance inst, HT ht) {
            this.observedClassDistribution.addToValue((int)inst.classValue(), inst.weight());
        }
    }

    public abstract static class LearningNode extends HT.Node {
        private static final long serialVersionUID = 1L;

        public LearningNode(double[] initialClassObservations) {
            super(initialClassObservations);
        }

        public abstract void learnFromInstance(Instance var1, HT var2);
    }

    public static class SplitNode extends HT.Node {
        private static final long serialVersionUID = 1L;
        protected InstanceConditionalTest splitTest;
        protected AutoExpandVector<HT.Node> children;

        public int calcByteSize() {
            return super.calcByteSize() + (int)(SizeOf.sizeOf(this.children) + SizeOf.fullSizeOf(this.splitTest));
        }

        public int calcByteSizeIncludingSubtree() {
            int byteSize = this.calcByteSize();
            Iterator var2 = this.children.iterator();

            while(var2.hasNext()) {
                HT.Node child = (HT.Node)var2.next();
                if (child != null) {
                    byteSize += child.calcByteSizeIncludingSubtree();
                }
            }

            return byteSize;
        }

        public SplitNode(InstanceConditionalTest splitTest, double[] classObservations, int size) {
            super(classObservations);
            this.splitTest = splitTest;
            this.children = new AutoExpandVector(size);
        }

        public SplitNode(InstanceConditionalTest splitTest, double[] classObservations) {
            super(classObservations);
            this.splitTest = splitTest;
            this.children = new AutoExpandVector();
        }

        public int numChildren() {
            return this.children.size();
        }

        public void setChild(int index, HT.Node child) {
            if (this.splitTest.maxBranches() >= 0 && index >= this.splitTest.maxBranches()) {
                throw new IndexOutOfBoundsException();
            } else {
                this.children.set(index, child);
            }
        }

        public HT.Node getChild(int index) {
            return (HT.Node)this.children.get(index);
        }

        public int instanceChildIndex(Instance inst) {
            return this.splitTest.branchForInstance(inst);
        }

        public boolean isLeaf() {
            return false;
        }

        public HT.FoundNode filterInstanceToLeaf(Instance inst, HT.SplitNode parent, int parentBranch) {
            int childIndex = this.instanceChildIndex(inst);
            if (childIndex >= 0) {
                HT.Node child = this.getChild(childIndex);
                return child != null ? child.filterInstanceToLeaf(inst, this, childIndex) : new HT.FoundNode((HT.Node)null, this, childIndex);
            } else {
                return new HT.FoundNode(this, parent, parentBranch);
            }
        }

        public void describeSubtree(HT ht, StringBuilder out, int indent) {
            for(int branch = 0; branch < this.numChildren(); ++branch) {
                HT.Node child = this.getChild(branch);
                if (child != null) {
                    StringUtils.appendIndented(out, indent, "if ");
                    out.append(this.splitTest.describeConditionForBranch(branch, ht.getModelContext()));
                    out.append(": ");
                    StringUtils.appendNewline(out);
                    child.describeSubtree(ht, out, indent + 2);
                }
            }

        }

        public int subtreeDepth() {
            int maxChildDepth = 0;
            Iterator var2 = this.children.iterator();

            while(var2.hasNext()) {
                HT.Node child = (HT.Node)var2.next();
                if (child != null) {
                    int depth = child.subtreeDepth();
                    if (depth > maxChildDepth) {
                        maxChildDepth = depth;
                    }
                }
            }

            return maxChildDepth + 1;
        }
    }

    public static class Node extends AbstractMOAObject {
        private static final long serialVersionUID = 1L;
        protected DoubleVector observedClassDistribution;

        public Node(double[] classObservations) {
            this.observedClassDistribution = new DoubleVector(classObservations);
        }

        public int calcByteSize() {
            return (int)(SizeOf.sizeOf(this) + SizeOf.fullSizeOf(this.observedClassDistribution));
        }

        public int calcByteSizeIncludingSubtree() {
            return this.calcByteSize();
        }

        public boolean isLeaf() {
            return true;
        }

        public HT.FoundNode filterInstanceToLeaf(Instance inst, HT.SplitNode parent, int parentBranch) {
            return new HT.FoundNode(this, parent, parentBranch);
        }

        public double[] getObservedClassDistribution() {
            return this.observedClassDistribution.getArrayCopy();
        }

        public double[] getClassVotes(Instance inst, HT ht) {
            return this.observedClassDistribution.getArrayCopy();
        }

        public boolean observedClassDistributionIsPure() {
            return this.observedClassDistribution.numNonZeroEntries() < 2;
        }

        public void describeSubtree(HT ht, StringBuilder out, int indent) {
            StringUtils.appendIndented(out, indent, "Leaf ");
            out.append(ht.getClassNameString());
            out.append(" = ");
            out.append(ht.getClassLabelString(this.observedClassDistribution.maxIndex()));
            out.append(" weights: ");
            this.observedClassDistribution.getSingleLineDescription(out, ht.treeRoot.observedClassDistribution.numValues());
            StringUtils.appendNewline(out);
        }

        public int subtreeDepth() {
            return 0;
        }

        public double calculatePromise() {
            double totalSeen = this.observedClassDistribution.sumOfValues();
            return totalSeen > 0.0D ? totalSeen - this.observedClassDistribution.getValue(this.observedClassDistribution.maxIndex()) : 0.0D;
        }

        public void getDescription(StringBuilder sb, int indent) {
            this.describeSubtree((HT)null, sb, indent);
        }
    }

    public static class FoundNode {
        public HT.Node node;
        public HT.SplitNode parent;
        public int parentBranch;

        public FoundNode(HT.Node node, HT.SplitNode parent, int parentBranch) {
            this.node = node;
            this.parent = parent;
            this.parentBranch = parentBranch;
        }
    }
}
