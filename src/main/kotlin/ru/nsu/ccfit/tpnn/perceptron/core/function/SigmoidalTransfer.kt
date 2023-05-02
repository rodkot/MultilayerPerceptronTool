package ru.nsu.ccfit.tpnn.perceptron.core.function

import ru.nsu.ccfit.tpnn.perceptron.core.TransferFunction
import kotlin.math.pow


class SigmoidalTransfer : TransferFunction {
    override fun evalute(value: Double): Double {
        return 1 / (1 + Math.E.pow(-value))
    }

    override fun evaluteDerivate(value: Double): Double {
        return value - value.pow(2.0)
    }
}


