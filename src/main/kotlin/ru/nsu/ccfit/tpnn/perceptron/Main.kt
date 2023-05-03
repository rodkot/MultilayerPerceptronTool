package ru.nsu.ccfit.tpnn.perceptron

import ru.nsu.ccfit.tpnn.perceptron.core.MultiLayerPerceptron
import ru.nsu.ccfit.tpnn.perceptron.core.function.SigmoidalTransfer
import ru.nsu.ccfit.tpnn.perceptron.core.utils.readData
import ru.nsu.ccfit.tpnn.perceptron.core.utils.minus

import org.jfree.chart.ChartFactory
import org.jfree.chart.ChartPanel
import org.jfree.chart.JFreeChart
import org.jfree.chart.plot.PlotOrientation
import org.jfree.data.category.DefaultCategoryDataset
import ru.nsu.ccfit.tpnn.perceptron.core.utils.calculateMiddle
import javax.swing.JFrame

fun main(args: Array<String>) {
    val inputSize = 18
    val outputSize = 2
    // Train MLP on loaded data
    val learningRate = 0.03
    val epochs = 29500
    val weightsLimit = -1.9 to 1.9;
    val biasLimit = -0.5 to 0.5;
    val layers = intArrayOf(18, 70, 63, 2)

    val dataTrain = readData(
        inputSize,
        outputSize,
        "C:\\Users\\rodio\\Project\\TPNN\\labs\\multilayer-perceptron\\src\\main\\resources\\my_data_train.csv"
    )
    val dataTest = readData(
        inputSize,
        outputSize,
        "C:\\Users\\rodio\\Project\\TPNN\\labs\\multilayer-perceptron\\src\\main\\resources\\my_data_test.csv"
    )

    val net = MultiLayerPerceptron(layers, weightsLimit, biasLimit, learningRate, SigmoidalTransfer())

    val errorEpoch = mutableListOf<MutableList<Double?>>()
    for (epoch in 1..epochs) {
        var error = arrayOf<Array<Double?>>()
        for ((inputs, targets) in dataTrain) {
            error += net.backPropagate(inputs, targets)
        }

        errorEpoch += mutableListOf<Double?>()
        for (i in 0 until outputSize) {
            val middle = calculateMiddle(error.map { it[i] }.toTypedArray())
            errorEpoch[epoch - 1] += middle
        }
        println("Epoch $epoch")
    }
    plot("G_total", errorEpoch.map { it[0] }.toTypedArray())
    plot("КГФ", errorEpoch.map { it[1] }.toTypedArray())

    println("Learning completed!")

    /* Test */
    for ((inputs, targets) in dataTest) {
        val values = net.execute(inputs)

        println("Error at step is ${(values - targets).map { it.toString() }}")
    }

}


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
    val frame = JFrame("My Frame")
    frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
    frame.contentPane.add(chartPanel)
    frame.pack()
    frame.isVisible = true
}