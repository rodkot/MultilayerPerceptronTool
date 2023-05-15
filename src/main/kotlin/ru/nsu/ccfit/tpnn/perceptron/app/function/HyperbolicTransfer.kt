package ru.nsu.ccfit.tpnn.perceptron.app.function

import ru.nsu.ccfit.tpnn.perceptron.core.function.TransferFunction
import kotlin.math.pow
import kotlin.math.tanh


class HyperbolicTransfer : TransferFunction {
    override fun evalute(value: Double): Double {
        return tanh(value)
    }

    override fun evaluteDerivate(value: Double): Double {
        return 1 - value.pow(2.0)
    }
}

