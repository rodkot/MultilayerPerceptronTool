package ru.nsu.ccfit.tpnn.perceptron.core

interface TransferFunction {
    /**
     * Funzione di trasferimento
     * @param value Valore in input
     * @return Valore funzione
     */
    fun evalute(value: Double): Double


    /**
     * Funzione derivata
     * @param value Valore in input
     * @return Valore funzione derivata
     */
    fun evaluteDerivate(value: Double): Double
}