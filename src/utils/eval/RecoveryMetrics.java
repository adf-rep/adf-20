package utils.eval;

import javafx.util.Pair;
import utils.windows.WindowedValue;

import java.util.ArrayList;

public class RecoveryMetrics {
    private ArrayList<Pair<Double, Double>> durations = new ArrayList<>();
    private ArrayList<Double> maxLosses = new ArrayList<>();
    private ArrayList<WindowedValue> accDrifts = new ArrayList<>();
    private ArrayList<WindowedValue> accStable = new ArrayList<>();
    private ArrayList<WindowedValue> lowOptDrifts = new ArrayList<>();
    private ArrayList<WindowedValue> lowOptStable = new ArrayList<>();
    private int streamSize;

    public RecoveryMetrics(int driftsNum, int streamSize) {
        for (int i = 0; i < driftsNum; i++) {
            this.durations.add(new Pair<>(-1.0, -1.0));
        }
        this.streamSize = streamSize;
    }

    public void updateDuration(int timestamp, int driftNum, boolean start) {
        if (start && this.durations.get(driftNum).getKey() < 0) {
            this.durations.set(driftNum, new Pair<>((double)timestamp, -1.0));
        } else if (!start && this.durations.get(driftNum).getValue() < 0) {
            Pair<Double, Double> duration = this.durations.get(driftNum);
            this.durations.set(driftNum, new Pair<>(duration.getKey(), (double)timestamp));
        }
    }

    public boolean isDurationStartReady(int driftNum) {
        return this.durations.get(driftNum).getKey() > 0;
    }
    public boolean isDurationReady(int driftNum) {
        return (this.durations.get(driftNum).getKey() > 0 && this.durations.get(driftNum).getValue() > 0);
    }

    public void updateLoss(double loss, int driftNum) {
        if (this.maxLosses.size() == driftNum) {
            this.maxLosses.add(loss);
        } else if (this.maxLosses.get(driftNum) < loss) {
            this.maxLosses.set(driftNum, loss);
        }
    }

    public void updateAccDrifts(double acc, int driftNum) {
        if (this.accDrifts.size() == driftNum) {
            this.accDrifts.add(new WindowedValue(Integer.MAX_VALUE));
        }

        this.accDrifts.get(driftNum).add(acc);
    }

    public void updateAccStable(double acc, int driftNum) {
        if (this.accStable.size() == driftNum) {
            this.accStable.add(new WindowedValue(Integer.MAX_VALUE));
        }

        this.accStable.get(driftNum).add(acc);
    }

    public void updateLowOptDrifts(double diff, int driftNum) {
        if (this.lowOptDrifts.size() == driftNum) {
            this.lowOptDrifts.add(new WindowedValue(Integer.MAX_VALUE));
        }

        this.lowOptDrifts.get(driftNum).add(diff);
    }

    public void updateLowOptStable(double diff, int driftNum) {
        if (this.lowOptStable.size() == driftNum) {
            this.lowOptStable.add(new WindowedValue(Integer.MAX_VALUE));
        }

        this.lowOptStable.get(driftNum).add(diff);
    }

    public double getAverageDuration() {
        return this.durations.stream()
                .filter(val -> val.getKey() > 0 && val.getValue() > 0)
                .mapToDouble(val -> (val.getValue() - val.getKey()) / streamSize)
                .average()
                .orElse(Double.NaN);
    }

    public double getAverageMaxLoss() {
        return this.maxLosses.stream()
                .mapToDouble(val -> val)
                .average()
                .orElse(Double.NaN);
    }

    public double getAverageAccDrifts() {
        return this.getAverage(this.accDrifts);
    }

    public double getAverageAccStable() {
        return this.getAverage(this.accStable);
    }

    public double getStdAccDrifts() {
        return this.getStd(this.accDrifts);
    }

    public double getStdAccStable() {
        return this.getStd(this.accStable);
    }

    public double getAverageLowOptDrifts() {
        return this.getAverage(this.lowOptDrifts);
    }

    public double getAverageLowOptStable() {
        return this.getAverage(this.lowOptStable);
    }

    private double getAverage(ArrayList<WindowedValue> list) {
        return list.stream()
                .mapToDouble(val -> val.getAverage())
                .average()
                .orElse(Double.NaN);
    }

    private double getStd(ArrayList<WindowedValue> list) {
        return list.stream()
                .mapToDouble(val -> val.getStd())
                .average()
                .orElse(Double.NaN);
    }
}
