package ru.nsu.ccfit.tpnn.perceptron.app.regression

import ru.nsu.ccfit.tpnn.perceptron.app.function.SigmoidalTransfer

import org.jfree.chart.ChartFactory
import org.jfree.chart.ChartPanel
import org.jfree.chart.JFreeChart
import org.jfree.chart.plot.PlotOrientation
import org.jfree.data.category.DefaultCategoryDataset
import ru.nsu.ccfit.tpnn.perceptron.core.utils.*
import javax.swing.JFrame

fun main(args: Array<String>) {
    val inputSize = 18
    val outputSize = 2
    // Train MLP on loaded data
    val learningRate = 0.03
    val epochs = 1000
    val weightsLimit = -1.9 to 1.9;
    val biasLimit = -0.5 to 0.5;
    val layers = intArrayOf(18, 50, 50, 2)

    val dataTrain = readData(
        inputSize,
        outputSize,
        "C:\\Users\\rodio\\Project\\TPNN\\labs\\multilayer-perceptron\\src\\main\\resources\\borehole\\my_data_train.csv"
    )
    val dataTest = readData(
        inputSize,
        outputSize,
        "C:\\Users\\rodio\\Project\\TPNN\\labs\\multilayer-perceptron\\src\\main\\resources\\borehole\\my_data_test.csv"
    )

    val net = RegressionPerceptron(layers, weightsLimit, biasLimit, learningRate, SigmoidalTransfer())

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
    val testResult = mutableListOf<Array<Double>>()
    val test = mutableListOf<Array<Double?>>()
    val error = mutableListOf<Array<Double?>>()
    for ((inputs, targets) in dataTest) {
        val d = net.execute(inputs)
        val t = targets
        testResult +=  d
        test += t
        error += (d-t)
    }
    plot("test G_toal",error.map { it[0] }.toTypedArray())
    plot("test КГФ",error.map { it[1] }.toTypedArray())

    println("G_total mse: ${mse(test.map { it[0] }.toTypedArray(), testResult.map { it[0] }.toTypedArray())}")
    println("КГФ mse: ${mse(test.map { it[1] }.toTypedArray(), testResult.map { it[1] }.toTypedArray())}")

}

