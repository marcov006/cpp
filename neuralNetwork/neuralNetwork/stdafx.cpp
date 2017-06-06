// stdafx.cpp : source file that includes just the standard includes
// neuralNetwork.pch will be the pre-compiled header
// stdafx.obj will contain the pre-compiled type information

#include "stdafx.h"

// TODO: reference any additional headers you need in STDAFX.H
// and not in this file

double Neuron::eta = 0.15; // overall net training rate [0.0..1]
double Neuron::alpha = 0.5; // mulitplier of last weight change (momentum) [0.0..n] 
