package ru.nsu.ccfit.tpnn.perceptron.core.utils

import org.jfree.chart.ChartFactory
import org.jfree.chart.ChartFrame
import org.jfree.chart.ChartPanel
import org.jfree.chart.JFreeChart
import org.jfree.chart.axis.NumberAxis
import org.jfree.chart.plot.PlotOrientation
import org.jfree.data.Range
import org.jfree.data.category.DefaultCategoryDataset
import org.jfree.data.xy.XYSeries
import org.jfree.data.xy.XYSeriesCollection
import javax.swing.JFrame

fun plot(nameTarget: String, errors: Array<Double?>) {
    // Create a dataset with some data
    var data = arrayOf<Int>()
    var error = arrayOf<Double>()
    for (i in errors.indices) {
        if (errors[i] != null) {
            data += (i + 1)
            error += errors[i]!!
        }
    }

    val dataset = DefaultCategoryDataset()
    for (i in data.indices) {
        dataset.addValue(error[i], "Error", data[i])
    }

    // Create a chart based on the dataset
    val chart: JFreeChart = ChartFactory.createLineChart(
        "Learning Curve Target $nameTarget",  // chart title
        "Epoch",  // domain axis label
        "Error",  // range axis label
        dataset,  // data
        PlotOrientation.VERTICAL,  // orientation
        true,  // include legend
        true,  // tooltips
        false  // urls
    )

    // Create a panel to display the chart
    val chartPanel = ChartPanel(chart)

    // Create a frame to display the panel
    val frame = JFrame("My $nameTarget")
    frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
    frame.contentPane.add(chartPanel)
    frame.pack()
    frame.isVisible = true
}

fun roc(targetName: String, test: Array<Double?>, result: Array<Double?>) {
    // Create a series to hold the ROC data points
    val rocSeries = XYSeries("ROC")

    var boarder = 0.0
    while (boarder<=1.0) {
        val confusionMatrix = ConfusionMatrix(test, result, boarder)
        rocSeries.add(confusionMatrix.fpr(), confusionMatrix.tpr())
        boarder+=0.00001
    }

// Create a dataset and add the ROC series to it
    val dataset = XYSeriesCollection()
    dataset.addSeries(rocSeries)

// Create a chart with a number axis for both the x and y axis
    val chart: JFreeChart = ChartFactory.createXYLineChart(
        "ROC Curve for $targetName", // chart title
        "False Positive Rate", // x axis label
        "True Positive Rate", // y axis label
        dataset, // data
        PlotOrientation.VERTICAL, // orientation
        true, // include legend
        true, // tooltips
        false // urls
    )

// Customize the chart by setting the range of the x and y axis
    val domainAxis = chart.xyPlot.domainAxis as NumberAxis
    domainAxis.range = Range(0.0, 1.0)
    val rangeAxis = chart.xyPlot.rangeAxis as NumberAxis
    rangeAxis.range = Range(0.0, 1.0)

// Display the chart in a frame or panel
    val chartPanel = ChartPanel(chart)
    val frame = ChartFrame("ROC Curve for $targetName", chart)
    frame.contentPane = chartPanel
    frame.pack()
    frame.isVisible = true
}