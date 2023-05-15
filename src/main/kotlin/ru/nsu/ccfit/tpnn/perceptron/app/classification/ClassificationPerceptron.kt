package ru.nsu.ccfit.tpnn.perceptron.app.classification

import ru.nsu.ccfit.tpnn.perceptron.core.MultiLayerPerceptron
import ru.nsu.ccfit.tpnn.perceptron.core.function.TransferFunction

class ClassificationPerceptron(
    layers: IntArray, weightsLimit: Pair<Double, Double>, biasLimit: Pair<Double, Double>,
    learningRate: Double, functionActivation: TransferFunction
) : MultiLayerPerceptron(
    layers,
    weightsLimit, biasLimit, learningRate, functionActivation
) {
    override fun changeOutLayer(value: Double): Double {
        return value
    }

    override fun changeInLayer(value: Double?): Double {
        return value ?: 0.0
    }
}