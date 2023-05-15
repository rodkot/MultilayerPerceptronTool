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


class ConfusionMatrix() {
    var truePositive = 0
    var trueNegative = 0
    var falsePositive = 0
    var falseNegative = 0

    constructor(test: Array<Double?>, result: Array<Double?>, boarder: Double) : this() {
        val res = result.map { if (it!! > boarder) 1.0 else 0.0 }
        for (i in test.indices) {
            when (test[i]) {
                0.0 -> {
                    if (res[i] == 1.0) {
                        falseNegative += 1
                    } else {
                        trueNegative += 1
                    }
                }

                1.0 -> {
                    if (res[i] == 1.0) {
                        truePositive += 1
                    } else {
                        falsePositive += 1
                    }
                }
            }
        }
    }

    fun tpr(): Double {
        return truePositive.toDouble() / (truePositive + falseNegative)
    }

    fun fpr(): Double {
        return falsePositive.toDouble() / (falsePositive + trueNegative)
    }

    //доля правильных ответов алгоритма
    fun accuracy(): Double {
        return (truePositive + trueNegative).toDouble() / (truePositive + trueNegative + falseNegative + falsePositive)
    }


    fun precision(): Double {
        return truePositive.toDouble() / (truePositive + falsePositive)
    }

    fun recall(): Double {
        return truePositive.toDouble() / (truePositive + falseNegative)
    }

    fun f1(): Double {
        val recall = recall()
        val precision = precision()
        return 2 * recall * precision / (recall + precision)
    }
}


fun mse(test: Array<Double?>, result: Array<Double?>): Double {
    require(test.size == result.size) { "Input arrays must have the same size" }
    var sum = 0.0
    var count = 0
    for (i in test.indices) {
        val ti = test[i]
        val ri = result[i]
        if (ti != null && ri != null) {
            sum += (ti - ri) * (ti - ri)
            count++
        }
    }
    return if (count > 0) sum / count else 0.0
}
