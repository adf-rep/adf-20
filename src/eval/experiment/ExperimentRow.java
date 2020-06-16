package eval.experiment;
import framework.Framework;
import moa.classifiers.Classifier;
import moa.classifiers.trees.HoeffdingAdaptiveTree;

public class ExperimentRow {

    public Framework framework;
    public String label;
    public String subLabel;

    public ExperimentRow(Framework framework, String label) {
        this.framework = framework;
        this.label = label;
        this.subLabel = "";
    }

    public ExperimentRow(Framework framework, String label, String subLabel) {
        this.framework = framework;
        this.label = label;
        this.subLabel = subLabel;
    }

    public static Classifier getExperimentClassifier() {
//        OnlineSmoothBoost cls = new OnlineSmoothBoost();
        //RCD cls = new RCD();
//        AccuracyWeightedEnsemble cls = new AccuracyWeightedEnsemble();
//        System.out.println(cls.baseLearnerOption.getValueAsCLIString());
//        AdaptiveRandomForest cls = new AdaptiveRandomForest();
//        OzaBag cls = new OzaBag();

//        cls.learnerOption = new ClassOption("baseLearner", 'l', "Classifier to train.", Classifier.class, "functions.Perceptron");

//        kNN cls = new kNN();
//        Perceptron cls = new Perceptron();

        HoeffdingAdaptiveTree cls = new HoeffdingAdaptiveTree();
        //RandomHoeffdingTree cls = new RandomHoeffdingTree();
        //RCD cls = new RCD();
        //AccuracyWeightedEnsemble cls = new AccuracyWeightedEnsemble();
        //cls.learnerOption = new ClassOption("baseLearner", 'l', "Classifier to train.", Classifier.class, "functions.Perceptron");

//        Iadem2 cls = new Iadem2();
//        NaiveBayes cls = new NaiveBayes();
        //SGDMultiClass cls = new SGDMultiClass();
        //DynamicNaiveBayes cls = new DynamicNaiveBayes(ForgettingStrategy.FIXED, 0.99);

        return cls;
    }
}
