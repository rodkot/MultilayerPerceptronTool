package ru.nsu.ccfit.tpnn.perceptron.core

import kotlin.math.abs

class MultiLayerPerceptron(
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


    fun execute(input: Array<Double?>): Array<Double> {
        var i: Int
        var j: Int
        var newValue: Double
        val output = DoubleArray(layers[layers.size - 1].length)

        // Put input
        i = 0
        while (i < layers[0].length) {
            layers[0].neurons[i].value = input[i] ?: 0.0
            i++
        }

        // Execute - hiddens + output
        var k = 1
        while (k < layers.size) {
            i = 0
            while (i < layers[k].length) {
                newValue = 0.0
                j = 0
                while (j < layers[k - 1].length) {
                    newValue += layers[k].neurons[i].weights[j] * layers[k - 1].neurons[j].value
                    j++
                }
                newValue += layers[k].neurons[i].bias
                layers[k].neurons[i].value = functionActivation.evalute(newValue)
                i++
            }
            k++
        }


        // Get output
        i = 0
        while (i < layers[layers.size - 1].length) {
            output[i] = layers[layers.size - 1].neurons[i].value
            i++
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

            // Calcolo l'errore dello strato corrente e ricalcolo i delta
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

            // Aggiorno i pesi dello strato successivo
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