package ru.nsu.ccfit.tpnn.perceptron

import com.github.doyaaaaaken.kotlincsv.dsl.csvReader
import java.io.File

val input_size = 15
val first_layer_size = 70
val second_layer_size = 63
val output_layer_size = 2

val epoch_count = 29500;
val learning_rate = 0.03;
val low_weight = -1.9;
val high_weight = 1.9;

fun main(args: Array<String>) {
    val inputSize = 18
    val hiddenSizes = listOf(32, 16)
    val outputSize = 2

    val mlp = MLP(inputSize, hiddenSizes, outputSize)



// Load CSV data
    val data = csvReader().readAll(File("C:\\Users\\rodio\\Project\\TPNN\\BoreholeProject\\my_data.csv")).map { row ->
        val inputs = row.toTypedArray().sliceArray(0 until inputSize).map { if (it.isEmpty()) 3.0 else it.toDouble() }.toDoubleArray()
        val targets =
            row.toTypedArray().sliceArray(inputSize until inputSize + outputSize).map {  if (it.isEmpty()) 3.0 else it.toDouble()  }.toDoubleArray()
        inputs to targets
    }

// Train MLP on loaded data
    val learningRate = 0.01
    val epochs = 100

    for (epoch in 1..epochs) {
        val error = 0.0
        for ((inputs, targets) in data) {
            mlp.backward(inputs, targets, learningRate)
        }
        println("Epoch $epoch: error=$error")
    }

}