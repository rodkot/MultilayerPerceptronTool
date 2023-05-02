package ru.nsu.ccfit.tpnn.perceptron.core.utils

import com.github.doyaaaaaken.kotlincsv.dsl.csvReader
import java.io.File


fun readData(inputSize: Int, targetSize: Int, path: String): List<Pair<Array<Double?>, Array<Double?>>> {
    return csvReader().readAll(File(path)).map { row ->
        val inputs =
            row.toTypedArray().sliceArray(0 until inputSize).map { if (it.isEmpty()) null else it.toDouble() }
                .toTypedArray()
        val targets =
            row.toTypedArray().sliceArray(inputSize until inputSize + targetSize)
                .map { if (it.isEmpty()) null else it.toDouble() }.toTypedArray()
        inputs to targets
    }
}