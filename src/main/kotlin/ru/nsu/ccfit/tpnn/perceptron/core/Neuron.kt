package ru.nsu.ccfit.tpnn.perceptron.core

import kotlin.random.Random

class Neuron(prevLayerSize: Int, weightsLimit: Pair<Double, Double>,biasLimit: Pair<Double, Double>) {
    var value = 0.0
    var weights: DoubleArray
    var bias = 0.0
    var delta = 0.0

    init {
        weights = DoubleArray(prevLayerSize)
        bias = Random.nextDouble(biasLimit.first, biasLimit.second)
        for (i in weights.indices) weights[i] = Random.nextDouble(weightsLimit.first, weightsLimit.second)
    }
}