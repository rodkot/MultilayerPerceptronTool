package ru.nsu.ccfit.tpnn.perceptron.core

class Layer(size: Int, weightsLimit: Pair<Double, Double>, biasLimit: Pair<Double, Double>, prevSize: Int) {
    var neurons = arrayOf<Neuron>()
    var length = size

    init {
        for (j in 0 until length) neurons += Neuron(prevSize, weightsLimit, biasLimit)
    }
}