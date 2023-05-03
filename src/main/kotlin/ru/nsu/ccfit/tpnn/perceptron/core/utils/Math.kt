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

fun calculateMiddle(arr: Array<Double?>): Double? {
    var sum: Double = 0.0
    var count: Int = 0

    for (element in arr) {
        if (element != null) {
            sum += element
            count++
        }
    }

    return if (count > 0) sum / count else null
}
