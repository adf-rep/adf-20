package output;

import java.io.*;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import eval.experiment.ExperimentResult;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.data.general.Dataset;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;


public abstract class Visualizer extends ApplicationFrame {

    private String title = "";
    private String dirPath = "";

    public Visualizer(final String name, final String title, String dirPath) {
        super(name);
        this.title = title;
        this.dirPath = dirPath;
    }

    public void plot(final List<ExperimentResult> results, int logGranularity, boolean show) {
        final HashMap<String, Dataset> datasets = createDatasets(results, logGranularity);
        for (Map.Entry<String, Dataset> entry : datasets.entrySet()) {
            final JFreeChart chart = createChart(entry.getValue(), title);
            this.writePlot(chart, entry.getKey());
            this.pack();
            RefineryUtilities.centerFrameOnScreen(this);
            this.setVisible(show);
        }
    }

    public void plot(String filepath, String fileName, boolean show) {
        final Dataset dataset = createDataset(filepath);
        final JFreeChart chart = createChart(dataset, title);
        this.writePlot(chart, fileName);
        this.pack();
        RefineryUtilities.centerFrameOnScreen(this);
        this.setVisible(show);
    }

    private void writePlot(JFreeChart chart, String fileName) {
        if (dirPath != null) {
            try {
                new File(dirPath).mkdirs();
                OutputStream out = new FileOutputStream(dirPath + "/" + fileName + ".png");
                ChartUtilities.writeChartAsPNG(out, chart, 1260, 760);
            } catch(IOException e) {
                e.printStackTrace();
            }
        }

        final ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(1000, 540));
        setContentPane(chartPanel);
    }

    abstract public HashMap<String, Dataset> createDatasets(final List<ExperimentResult> results, int logGranularity);

    abstract public Dataset createDataset(final String filepath);

    abstract public JFreeChart createChart(final Dataset dataset, final String title);
}
