package ru.nsu.ccfit.tpnn.perceptron

import kotlin.math.exp

class MLP(private val inputSize: Int, private val hiddenSizes: List<Int>, private val outputSize: Int) {
    private val layers: MutableList<Layer> = mutableListOf()

    init {
        // Create the layers of the MLP
        var prevSize = inputSize
        for (hiddenSize in hiddenSizes) {
            layers.add(Layer(prevSize, hiddenSize))
            prevSize = hiddenSize
        }
        layers.add(Layer(prevSize, outputSize))
    }

    fun forward(inputData: DoubleArray): DoubleArray {
        var output = inputData
        for (layer in layers) {
            output = layer.forward(output)
        }
        return output
    }

    fun backward(inputData: DoubleArray, targetOutput: DoubleArray, learningRate: Double) {
        var outputError = targetOutput - forward(inputData)
        for (i in layers.size - 1 downTo 0) {
            val layer  = layers[i]
            outputError = layer.backward(outputError, learningRate)
        }
    }

    inner class Layer(private val inputSize: Int, private val outputSize: Int) {
        private val weights: Array<DoubleArray> = Array(inputSize) { DoubleArray(outputSize) { Math.random() } }
        private val bias: DoubleArray = DoubleArray(outputSize)

        private lateinit var output: DoubleArray

        fun forward(inputData: DoubleArray): DoubleArray {
            val z = DoubleArray(outputSize)
            for (j in 0 until outputSize) {
                for (i in 0 until inputSize) {
                    z[j] += inputData[i] * weights[i][j]
                }
                z[j] += bias[j]
            }
            output = sigmoid(z)
            return output
        }

        fun backward(outputError: DoubleArray, learningRate: Double): DoubleArray {
            val delta = outputError * sigmoidDerivative(output)
            val weightError = Array(inputSize) { DoubleArray(outputSize) }
            for (i in 0 until inputSize) {
                for (j in 0 until outputSize) {
                    weightError[i][j] = delta[j] * layers[layers.indexOf(this) - 1].weights[j][i]
                }
            }
            val biasError = delta
            val inputError = DoubleArray(inputSize)
            for (i in 0 until inputSize) {
                for (j in 0 until outputSize) {
                    inputError[i] += delta[j] * weights[i][j]
                }
            }
            for (i in 0 until outputSize) {
                for (j in 0 until outputSize) {
                    weights[i][j] += learningRate * output[i] * delta[j]
                }
            }
            for (j in 0 until outputSize) {
                bias[j] += learningRate * biasError[j]
            }
            return inputError
        }

        private fun sigmoid(x: DoubleArray): DoubleArray {
            return DoubleArray(x.size) { i -> 1.0 / (1.0 + exp(-x[i])) }
        }

        private fun sigmoidDerivative(x: DoubleArray): DoubleArray {
            return DoubleArray(x.size) { i -> x[i] * (1.0 - x[i]) }
        }
    }
}
