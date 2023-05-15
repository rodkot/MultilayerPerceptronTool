package ru.nsu.ccfit.tpnn.perceptron.app.classification

import ru.nsu.ccfit.tpnn.perceptron.app.function.SigmoidalTransfer

import ru.nsu.ccfit.tpnn.perceptron.core.utils.*

fun main(args: Array<String>) {
    val inputSize = 21
    val outputSize = 1
    // Train MLP on loaded data
    val learningRate = 0.09
    val epochs = 5
    val weightsLimit = -1.9 to 1.9;
    val biasLimit = -0.5 to 0.5;
    val boarder = 0.5

    val layers = intArrayOf(inputSize, 50, 90, outputSize)

    val dataTrain = readData(
        inputSize,
        outputSize,
        "C:\\Users\\rodio\\Project\\TPNN\\labs\\multilayer-perceptron\\src\\main\\resources\\mushrooms\\my_data_train.csv"
    )
    val dataTest = readData(
        inputSize,
        outputSize,
        "C:\\Users\\rodio\\Project\\TPNN\\labs\\multilayer-perceptron\\src\\main\\resources\\mushrooms\\my_data_test.csv"
    )

    val net = ClassificationPerceptron(layers, weightsLimit, biasLimit, learningRate, SigmoidalTransfer())

    var errorEpoch = arrayOf<Double?>()
    for (epoch in 1..epochs) {
        var error = arrayOf<Double?>()
        for ((inputs, targets) in dataTrain) {
            error += net.backPropagate(inputs, targets)[0]
        }

        val middle = calculateMiddle(error)
        errorEpoch += middle

        println("Epoch $epoch")
    }
    plot("is-poisonus", errorEpoch)


    println("Learning completed!")


    /* Test */


    var testResult = arrayOf<Double?>()
    var test = arrayOf<Double?>()
    var error = arrayOf<Double?>()

    for ((inputs, targets) in dataTest) {
        val exec = net.execute(inputs)[0]
        val res = if (exec > boarder) 1.0 else 0.0
        testResult += exec
        test += targets[0]

        error += (res - exec)
    }
    plot("test 'is-poisonus'", error)


    val confusionMatrix = ConfusionMatrix(test, testResult, boarder)

    println("'is-poisonus' accuracy: ${confusionMatrix.accuracy()}")
    println("'is-poisonus' precision: ${confusionMatrix.precision()}")
    println("'is-poisonus' recall: ${confusionMatrix.recall()}")
    println("'is-poisonus' f1: ${confusionMatrix.f1()}")

    roc("'is-poisonus'",test,testResult)


}


