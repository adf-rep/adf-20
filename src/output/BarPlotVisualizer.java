package output;
import eval.experiment.ExperimentResult;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.labels.StandardCategoryItemLabelGenerator;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.category.BarRenderer;
import org.jfree.chart.renderer.category.StandardBarPainter;
import org.jfree.chart.title.LegendTitle;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.general.Dataset;
import org.jfree.ui.RectangleEdge;

import java.awt.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class BarPlotVisualizer extends Visualizer {

    private boolean groupable = false;

    BarPlotVisualizer(final String name, final String title, String dirPath, boolean groupable) {
        super(name, title, dirPath);
        this.groupable = groupable;
    }

    @Override
    public HashMap<String, Dataset> createDatasets(final List<ExperimentResult> results, int logGranularity) {
        HashMap<String, Dataset> datasets = new HashMap<>();
        String[] outputNames = RaWritter.getOutputNames(results).get("averages");

        for (String outputName : outputNames) {
            final DefaultCategoryDataset dataset = new DefaultCategoryDataset();
            List<Double> data = new ArrayList<>();
            List<String> labels = new ArrayList<>();
            List<String> subLabels = new ArrayList<>();

            for (ExperimentResult result : results) {
                if (result.averageMeasurements.get(outputName) == null) continue;
                data.add(result.averageMeasurements.get(outputName));
                labels.add(result.label);
                subLabels.add(result.subLabel);
            }

            for (int i = 0; i < data.size(); i++) {
                dataset.addValue(data.get(i), labels.get(i), subLabels.get(i));
            }

            datasets.put(outputName, dataset);
        }

        return datasets;
    }

    @Override
    public Dataset createDataset(String filepath) {
        final DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        try (BufferedReader br = new BufferedReader(new FileReader(filepath))) {

            String line;
            while ((line = br.readLine()) != null) {
                String[] content = line.split(",");
                double value = Double.parseDouble(content[1]);
                String[] naming = content[0].split("#");

                dataset.addValue(value, naming[0], naming.length == 2 ? naming[1] : "");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return dataset;
    }

    @Override
    public JFreeChart createChart(final Dataset dataset, final String title) {
        final JFreeChart chart = ChartFactory.createBarChart(
                title,
                this.groupable ? "Budget" : "",
                "Value",
                (DefaultCategoryDataset) dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );

        chart.setBackgroundPaint(Color.white);

        final CategoryPlot plot = chart.getCategoryPlot();
        plot.setBackgroundPaint(Color.white);
        plot.setDomainGridlinePaint(Color.white);
        plot.setRangeGridlinePaint(Color.white);

        final BarRenderer renderer = (BarRenderer) plot.getRenderer();
        renderer.setBarPainter(new StandardBarPainter());
        renderer.setSeriesPaint(0, Color.RED);
        renderer.setSeriesPaint(1, Color.BLUE);
        renderer.setSeriesPaint(2, Color.GREEN);
        renderer.setSeriesPaint(3, Color.ORANGE);
        renderer.setSeriesPaint(4, Color.CYAN);
        renderer.setSeriesPaint(5, Color.MAGENTA);
        renderer.setSeriesPaint(6, Color.DARK_GRAY);

        renderer.setIncludeBaseInRange(false);
        renderer.setBaseItemLabelsVisible(true);
        renderer.setBaseItemLabelPaint(Color.BLACK);
        renderer.setMaximumBarWidth(0.1);
        renderer.setItemMargin(0.05);
        renderer.setBaseItemLabelGenerator(new StandardCategoryItemLabelGenerator("{2}", NumberFormat.getInstance()));
        plot.setRenderer(renderer);

        final NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        //rangeAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
        rangeAxis.setAutoRangeIncludesZero(false);
        rangeAxis.setUpperMargin(0.1);
        rangeAxis.setLowerMargin(0.25);

        LegendTitle legend = chart.getLegend();
        legend.setPosition(RectangleEdge.RIGHT);

        return chart;
    }
}
