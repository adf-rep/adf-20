package utils.windows;

import java.util.ArrayList;

public class IncrementalValue {

    protected double average = 0.0;
    protected double varSum = 0.0;
    protected double sum = 0.0;
    protected int n = 0;
    protected boolean negFix = true;

    public IncrementalValue() {}

    public void add(double value) {
        this.update(value);
        if (value < 0.0) this.negFix = false;
    }

    private void update(double value) {
        double oldAverage = this.average;
        this.n++;
        this.sum += value;
        this.average += ((value - this.average) / n);
        this.varSum = this.varSum + (value - this.average) * (value - oldAverage);

        if (negFix && this.average < 0.0) this.average = 0.0; // approximation error
    }

    public double getSum() {
        return this.sum;
    }

    public double getAverage() {
        return this.average;
    }

    public double getStd() {
        return Math.sqrt(this.varSum / this.n);
    }

}
