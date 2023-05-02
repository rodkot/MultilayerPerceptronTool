package ru.nsu.ccfit.tpnn.perceptron.core.utils

operator fun Array<Double>.minus(targets: Array<Double?>): Array<Double?> {
    if (this.size != targets.size)
        throw Exception("Не совпадают размерности")
    var result = arrayOf<Double?>()
    for (i in targets.indices) {
        if (targets[i] == null)
            result += null
        else
            result += (this[i] - targets[i]!!)
    }
    return result
}