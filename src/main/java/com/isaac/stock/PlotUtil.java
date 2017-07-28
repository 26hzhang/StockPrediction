package com.isaac.stock;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.NumberTickUnit;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;

public class PlotUtil {

	public static void plot (double[] predicts, double[] actuals) {
		double[] index = new double[predicts.length];
		for (int i = 0; i < predicts.length; i++)
			index[i] = i;
		final XYSeriesCollection dataSet = new XYSeriesCollection();
		addSeries(dataSet, index, predicts, "Predicts");
		addSeries(dataSet, index, actuals, "Actuals");
		final JFreeChart chart = ChartFactory.createXYLineChart(
				"Prediction Result", // chart title
				"Index", // x axis label
				"Stock Close Price", // y axis label
				dataSet, // data
				PlotOrientation.VERTICAL,
				true, // include legend
				true, // tooltips
				false // urls
		);
		XYPlot xyPlot = chart.getXYPlot();
		// X-axis
		final NumberAxis domainAxis = (NumberAxis) xyPlot.getDomainAxis();
		domainAxis.setRange(0, 160);
		domainAxis.setTickUnit(new NumberTickUnit(20));
		domainAxis.setVerticalTickLabels(true);
		// Y-axis
		final NumberAxis rangeAxis = (NumberAxis) xyPlot.getRangeAxis();
		rangeAxis.setRange(600, 850);
		rangeAxis.setTickUnit(new NumberTickUnit(50));
		final ChartPanel panel = new ChartPanel(chart);
		final JFrame f = new JFrame();
		f.add(panel);
		f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
		f.pack();
		f.setVisible(true);
	}

	private static void addSeries(final XYSeriesCollection dataSet, double[] x, double[] y, final String label){
		final XYSeries s = new XYSeries(label);
		for( int j=0; j<x.length; j++ ) s.add(x[j],y[j]);
		dataSet.addSeries(s);
	}

}