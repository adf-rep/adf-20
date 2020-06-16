package utils.eval;

import java.util.ArrayList;

public class ResourcesMetrics {

    private ArrayList<Double> classificationTimeMeasurements;
    private ArrayList<Double> updateTimeMeasurements;
    private ArrayList<Double> memoryUsageMeasurements;
    private double allocationMemoryUsageMeasurement;

    private long startTime;
    private long startMemory;

    public ResourcesMetrics() {
        this.classificationTimeMeasurements = new ArrayList<>();
        this.updateTimeMeasurements = new ArrayList<>();
        this.memoryUsageMeasurements = new ArrayList<>();
    }

    public void startTimer() {
        this.startTime = System.nanoTime();
    }

    public long stopTimer() {
        return (System.nanoTime() - this.startTime) / 1000;
    }

    public void addClassificationTimeMeasurement(double classificationTimeMs) {
        this.classificationTimeMeasurements.add(classificationTimeMs);
    }

    public void addUpdateTimeMeasurement(double updateTimeMs) {
        this.updateTimeMeasurements.add(updateTimeMs);
    }

    public void addMemoryUsageMeasurement(double runningMemoryMeasurementBytes) {
        this.memoryUsageMeasurements.add(runningMemoryMeasurementBytes);
    }

    public void startMem() {
        this.startMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
    }

    public long stopMem() {
        long endMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        return (endMemory - this.startMemory) / 1000000;
    }

    static public long getMemoryUsage() {
        return Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
    }

    public ArrayList<Double> getClassificationTimeMeasurements() {
        return this.classificationTimeMeasurements;
    }

    public double getAvgClassificationTime() {
        return this.classificationTimeMeasurements.stream().mapToDouble(val -> val).average().orElse(Double.NaN);
    }

    public ArrayList<Double> getUpdateTimeMeasurements() {
        return this.updateTimeMeasurements;
    }

    public double getAvgUpdateTime() {
        return this.updateTimeMeasurements.stream().mapToDouble(val -> val).average().orElse(Double.NaN);
    }

    public ArrayList<Double> getMemoryUsageMeasurements() {
        return this.memoryUsageMeasurements;
    }

    public double getAvgMemoryUsage() {
        return this.memoryUsageMeasurements.stream().mapToDouble(val -> val).average().orElse(Double.NaN);
    }

    public ResourcesMetrics setAllocationMemoryMeasurement(double allocationMemoryUsageMeasurement) {
        this.allocationMemoryUsageMeasurement = allocationMemoryUsageMeasurement;
        return this;
    }

    public double getAllocationMemoryUsageMeasurement() {
        return this.allocationMemoryUsageMeasurement;
    }

}
