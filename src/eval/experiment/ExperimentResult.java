package eval.experiment;
import output.writer.OutputWriter;

import java.util.HashMap;
import java.util.List;

public class ExperimentResult {

    public HashMap<String, List<Double>> seriesMeasurements;
    public HashMap<String, Double> averageMeasurements;
    public HashMap<String, List<OutputWriter>> other;
    public double queriesFactor;
    public String label;
    public String subLabel;

    public static final String ACCURACY = "accuracy";
    public static final String PRECISION = "precision";
    public static final String RECALL = "recall";
    public static final String F1 = "f1";
    public static final String KAPPA = "kappa";
    public static final String KAPPA_T = "kappa_t";
    public static final String AUC = "auc";
    public static final String AUC_SCORED = "auc_scored";
    public static final String G_MEAN_SERIES = "g_mean_series";
    public static final String G_MEAN = "g_mean";

    public static final String CLASSIFICATION_TIME_SERIES = "classification_time_series";
    public static final String CLASSIFICATION_TIME = "classification_time";
    public static final String UPDATE_TIME_SERIES = "update_time_series";
    public static final String UPDATE_TIME = "update_time";
    public static final String MEMORY_USAGE_SERIES = "memory_usage_series";
    public static final String MEMORY_USAGE = "memory_usage";
    public static final String ALLOCATION_MEMORY_USAGE = "allocation_memory_usage";

    public static final String ACCURACY_SERIES = "accuracy_series";
    public static final String ACCURACY_SERIES_10 = "accuracy_series_10";
    public static final String ACCURACY_SERIES_100 = "accuracy_series_100";
    public static final String ACCURACY_SERIES_1000 = "accuracy_series_1000";
    public static final String ACCURACY_SERIES_10000 = "accuracy_series_10000";

    public static final String ACCURACY_SERIES_ADWIN = "accuracy_series_adwin";
    public static final String ACCURACY_SERIES_ADWIN_001 = "accuracy_series_adwin_001";
    public static final String ACCURACY_SERIES_ADWIN_005 = "accuracy_series_adwin_005";
    public static final String ACCURACY_SERIES_ADWIN_01 = "accuracy_series_adwin_01";
    public static final String ACCURACY_SERIES_ADWIN_02 = "accuracy_series_adwin_02";

    public static final String KAPPA_SERIES = "kappa_series";
    public static final String KAPPA_T_SERIES = "kappa_t_series";
    public static final String KAPPA_SERIES_10 = "kappa_series_10";
    public static final String KAPPA_SERIES_100 = "kappa_series_100";
    public static final String KAPPA_SERIES_1000 = "kappa_series_1000";
    public static final String KAPPA_SERIES_10000 = "kappa_series_10000";

    public static final String KAPPA_SERIES_ADWIN = "kappa_series_adwin";
    public static final String KAPPA_T_SERIES_ADWIN = "kappa_t_series_adwin";
    public static final String KAPPA_SERIES_ADWIN_001 = "kappa_series_adwin_001";
    public static final String KAPPA_SERIES_ADWIN_005 = "kappa_series_adwin_005";
    public static final String KAPPA_SERIES_ADWIN_01 = "kappa_series_adwin_01";
    public static final String KAPPA_SERIES_ADWIN_02 = "kappa_series_adwin_02";

    public static final String PREDICTIONS = "predictions";

    public ExperimentResult(String label, String subLabel) {
        this.seriesMeasurements = new HashMap<>();
        this.averageMeasurements = new HashMap<>();
        this.other = new HashMap<>();
        this.queriesFactor = 0.0;
        this.label = label;
        this.subLabel = subLabel;
    }
}
