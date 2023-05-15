package ru.nsu.ccfit.tpnn.perceptron.core

import ru.nsu.ccfit.tpnn.perceptron.core.function.TransferFunction
import kotlin.math.abs

abstract class MultiLayerPerceptron(
    layers: IntArray,
    weightsLimit: Pair<Double, Double>,
    biasLimit: Pair<Double, Double>,
    private var learningRate: Double,
    private var functionActivation: TransferFunction
) : Cloneable {

    private var layers = listOf<Layer>()

    init {
        for (i in layers.indices) {
            if (i != 0) {
                this.layers += Layer(layers[i], weightsLimit, biasLimit, layers[i - 1])
            } else {
                this.layers += Layer(layers[i], weightsLimit, biasLimit, 0)
            }
        }
    }

    abstract fun changeOutLayer(value: Double): Double
    abstract fun changeInLayer(value: Double?): Double

    fun execute(input: Array<Double?>): Array<Double> {

        var newValue: Double
        val output = DoubleArray(layers[layers.size - 1].length)

        // Put input
        for (i in 0 until layers[0].length) {
            layers[0].neurons[i].value = changeInLayer(input[i])
        }

        // Execute - hiddens + output
        for (k in 1 until layers.size) {
            for (i in 0 until layers[k].length) {
                newValue = 0.0
                for (j in 0 until layers[k - 1].length) {
                    newValue += layers[k].neurons[i].weights[j] * layers[k - 1].neurons[j].value
                }
                newValue += layers[k].neurons[i].bias
                layers[k].neurons[i].value = functionActivation.evalute(newValue)

            }

        }

        for (i in 0 until layers[layers.size - 1].length) {
            output[i] = changeOutLayer(layers[layers.size - 1].neurons[i].value)
        }

        return output.toTypedArray()
    }


    fun backPropagate(input: Array<Double?>, output: Array<Double?>): Array<Double?> {
        val newOutput = execute(input)
        var error: Double
        var i: Int
        var j: Int

        /* doutput = correct output (output) */

        i = 0
        while (i < layers[layers.size - 1].length) {
            if (output[i] != null) {
                error = output[i]!! - newOutput[i]
                layers[layers.size - 1].neurons[i].delta = error * functionActivation.evaluteDerivate(newOutput[i])
            }
            i++
        }
        var k: Int = layers.size - 2
        while (k >= 0) {

            i = 0
            while (i < layers[k].length) {
                error = 0.0
                j = 0
                while (j < layers[k + 1].length) {
                    error += layers[k + 1].neurons[j].delta * layers[k + 1].neurons[j].weights[i]
                    j++
                }
                layers[k].neurons[i].delta =
                    error * functionActivation.evaluteDerivate(layers[k].neurons[i].value)
                i++
            }

            i = 0
            while (i < layers[k + 1].length) {
                j = 0
                while (j < layers[k].length) {
                    layers[k + 1].neurons[i].weights[j] += learningRate * layers[k + 1].neurons[i].delta * layers[k].neurons[j].value
                    j++
                }
                layers[k + 1].neurons[i].bias += learningRate * layers[k + 1].neurons[i].delta
                i++
            }
            k--
        }

        var listError = arrayOf<Double?>()
        i = 0
        while (i < output.size) {
            if (output[i] == null) {
                listError += null
            } else {
                listError += abs(newOutput[i] - output[i]!!)
            }
            i++
        }
        return listError
    }
}