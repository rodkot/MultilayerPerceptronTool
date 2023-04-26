package ru.nsu.ccfit.tpnn.perceptron

operator fun DoubleArray.minus(other: DoubleArray): DoubleArray {
    require(size == other.size) { "Arrays must have the same size" }
    return DoubleArray(size) { i -> this[i] - other[i] }
}

operator fun DoubleArray.times(other: DoubleArray): DoubleArray {
    require(size == other.size) { "Arrays must have the same size" }
    return DoubleArray(size) { i -> this[i] * other[i] }
}
