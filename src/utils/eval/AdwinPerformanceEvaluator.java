package utils.eval;

import moa.classifiers.core.driftdetection.ADWIN;
import moa.evaluation.BasicClassificationPerformanceEvaluator;

public class AdwinPerformanceEvaluator extends BasicClassificationPerformanceEvaluator {
    private static final long serialVersionUID = 1L;
    private double delta = 0.002;

    public AdwinPerformanceEvaluator(double delta) {
        this.delta = delta;
    }

    public AdwinPerformanceEvaluator() {
    }

    protected BasicClassificationPerformanceEvaluator.Estimator newEstimator() {
        return new AdwinPerformanceEvaluator.AdwinEstimator(this.delta);
    }

    public class AdwinEstimator implements BasicClassificationPerformanceEvaluator.Estimator {
        protected ADWIN adwin = new ADWIN();
        private double delta;

        public AdwinEstimator(double delta) {
            this.delta = delta;
        }

        public void add(double value) {
            this.adwin.setInput(value, this.delta);
        }

        public double estimation() {
            return this.adwin.getEstimation();
        }
    }
}
