package utils.windows;

import utils.ForgettingStrategy;

import java.util.HashMap;

public class ForgettingEstimator {
    protected double count;
    private double ls;
    private double ss;

    private ForgettingStrategy forgettingStrategy;
    private int timestamp;
    private double decay;

    public ForgettingEstimator(ForgettingStrategy forgettingStrategy, double decay) {
        this.forgettingStrategy = forgettingStrategy;
        this.decay = decay;
        this.count = 0.0;
        this.ls = 0.0;
        this.ss = 0.0;
    }

    public void update(double v, HashMap<String, Double> indicators, int t) {
        double decay = this.forgettingStrategy(indicators);
        double d = Math.pow(decay, t - this.timestamp);

        this.count = d * this.count + 1.0;
        this.ls = d * this.ls + v;
        this.ss = d * this.ss + Math.pow(v, 2.0);
        this.timestamp = t;
    }

    private double forgettingStrategy(HashMap<String, Double> indicators) {
        switch(this.forgettingStrategy) {
            case FIXED: return this.decay;
            case ERROR_DRIVEN: return (1.0 - indicators.get("error"));
            case ERROR_DRIVEN_ADJ: return -0.1 * indicators.get("error") + 1.0;
            default: return 1.0;
        }
    }

    public double getCount() {
        return this.count;
    }

    public double getSum() {
        return this.ls;
    }

    public double getMean() {
        if (this.count == 0.0) return Double.NaN;
        return this.ls / this.count;
    }

    public double getVar() {
        if (this.count == 0.0) return Double.NaN;
        double var = this.ss / this.count - Math.pow(this.ls / this.count, 2.0);
        return Math.max(var, 0.0);
    }

    public double getStd() {
        if (this.count == 0.0) return Double.NaN;
        double var = this.ss / this.count - Math.pow(this.ls / this.count, 2.0);
        return Math.sqrt(Math.max(var, 0.0));
    }

    public ForgettingEstimator copy() {
        ForgettingEstimator fe = new ForgettingEstimator(this.forgettingStrategy, this.decay);
        fe.count = this.count;
        fe.ls = this.ls;
        fe.ss = this.ss;
        fe.timestamp = this.timestamp;

        return fe;
    }

}
