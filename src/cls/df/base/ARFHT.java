package cls.df.base;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.bayes.NaiveBayes;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.core.Utils;

/* Disclaimer: This a copy of the original ARFHT classifier implemented in MOA. */

public class ARFHT extends HT {
    private static final long serialVersionUID = 1L;
    public IntOption subspaceSizeOption = new IntOption("subspaceSizeSize", 'k', "Number of features per subset for each node split. Negative values = #features - k", 2, -2147483648, 2147483647);

    public String getPurposeString() {
        return "Adaptive Random Forest Hoeffding Tree for data streams. Base learner for AdaptiveRandomForest.";
    }

    public ARFHT() {
        this.removePoorAttsOption = null;
    }

    protected LearningNode newLearningNode(double[] initialClassObservations) {
        int predictionOption = this.leafpredictionOption.getChosenIndex();
        Object ret;
        if (predictionOption == 0) {
            ret = new ARFHT.RandomLearningNode(initialClassObservations, this.subspaceSizeOption.getValue());
        } else if (predictionOption == 1) {
            ret = new ARFHT.LearningNodeNB(initialClassObservations, this.subspaceSizeOption.getValue());
        } else {
            ret = new ARFHT.LearningNodeNBAdaptive(initialClassObservations, this.subspaceSizeOption.getValue());
        }

        return (LearningNode)ret;
    }

    public boolean isRandomizable() {
        return true;
    }

    public static class LearningNodeNBAdaptive extends ARFHT.LearningNodeNB {
        private static final long serialVersionUID = 1L;
        protected double mcCorrectWeight = 0.0D;
        protected double nbCorrectWeight = 0.0D;

        public LearningNodeNBAdaptive(double[] initialClassObservations, int subspaceSize) {
            super(initialClassObservations, subspaceSize);
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

    public static class LearningNodeNB extends ARFHT.RandomLearningNode {
        private static final long serialVersionUID = 1L;

        public LearningNodeNB(double[] initialClassObservations, int subspaceSize) {
            super(initialClassObservations, subspaceSize);
        }

        public double[] getClassVotes(Instance inst, HT ht) {
            return this.getWeightSeen() >= (double)ht.nbThresholdOption.getValue() ? NaiveBayes.doNaiveBayesPrediction(inst, this.observedClassDistribution, this.attributeObservers) : super.getClassVotes(inst, ht);
        }

        public void disableAttribute(int attIndex) {
        }
    }

    public static class RandomLearningNode extends ActiveLearningNode {
        private static final long serialVersionUID = 1L;
        protected int[] listAttributes;
        protected int numAttributes;

        public RandomLearningNode(double[] initialClassObservations, int subspaceSize) {
            super(initialClassObservations);
            this.numAttributes = subspaceSize;
        }

        public void learnFromInstance(Instance inst, HT ht) {
            this.observedClassDistribution.addToValue((int)inst.classValue(), inst.weight());
            int j;
            int i;
            if (this.listAttributes == null) {
                this.listAttributes = new int[this.numAttributes];

                label51:
                for(j = 0; j < this.numAttributes; ++j) {
                    boolean isUnique = false;

                    while(true) {
                        while(true) {
                            if (isUnique) {
                                continue label51;
                            }

                            this.listAttributes[j] = ht.classifierRandom.nextInt(inst.numAttributes() - 1);
                            isUnique = true;

                            for(i = 0; i < j; ++i) {
                                if (this.listAttributes[j] == this.listAttributes[i]) {
                                    isUnique = false;
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            for(j = 0; j < this.numAttributes - 1; ++j) {
                int k = this.listAttributes[j];
                k = ARFHT.modelAttIndexToInstanceAttIndex(k, inst);
                AttributeClassObserver obs = (AttributeClassObserver)this.attributeObservers.get(k);
                if (obs == null) {
                    obs = inst.attribute(k).isNominal() ? ht.newNominalClassObserver() : ht.newNumericClassObserver();
                    this.attributeObservers.set(k, obs);
                }

                obs.observeAttributeClass(inst.value(k), (int)inst.classValue(), inst.weight());
            }

        }
    }
}