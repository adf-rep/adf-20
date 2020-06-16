package framework;

public class FrameworkUpdate {
    public boolean labeled;
    public boolean predictedCorrect;

    FrameworkUpdate(boolean labeled) {
        this.labeled = labeled;
    }

    FrameworkUpdate(boolean labeled, boolean predictedCorrect) {
        this.labeled = labeled;
        this.predictedCorrect = predictedCorrect;
    }

    public String toString() {
        return "#" + (this.labeled ? 1 : 0) + "" + (this.predictedCorrect ? 1 : 0);
    }

}
