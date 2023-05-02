package ru.nsu.ccfit.tpnn.perceptron

import com.github.doyaaaaaken.kotlincsv.dsl.csvReader
import ru.nsu.ccfit.tpnn.perceptron.core.MultiLayerPerceptron
import ru.nsu.ccfit.tpnn.perceptron.core.function.SigmoidalTransfer
import ru.nsu.ccfit.tpnn.perceptron.core.utils.readData
import ru.nsu.ccfit.tpnn.perceptron.core.utils.minus
import java.io.File

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

    for (epoch in 1..epochs) {
        for ((inputs, targets) in dataTrain) {
            net.backPropagate(inputs, targets)
        }
        println("Epoch $epoch")
    }

    println("Learning completed!")

    /* Test */
    for ((inputs, targets) in dataTest) {
        val values = net.execute(inputs)

        println("Error at step is ${(values - targets).map { it.toString() }}")
    }

}


