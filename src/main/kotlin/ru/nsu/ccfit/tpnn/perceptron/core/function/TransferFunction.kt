package ru.nsu.ccfit.tpnn.perceptron.core.function

interface TransferFunction {
    //function activation
    fun evalute(value: Double): Double

    //Derived function
    fun evaluteDerivate(value: Double): Double
}