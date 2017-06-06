#include "stdafx.h"

void Neuron::updateInputWeights(Layer &prevLayer) 
{
	// The weights to be updated are in the Connection container
	// in the neurons in the preceding layer

	// cout << "updateInputWeights m_myIndex = " << m_myIndex << endl;

	for (unsigned n = 0; n < prevLayer.size(); n++) {
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

		double newDeltaWeight = 
			//Individual input, magnified by the gradient and train rate:
			eta
			* neuron.getOutputVal()
			* m_gradient
			// Also add momentum = a fraction of the previous delta weight
			+ alpha
			* oldDeltaWeight;

		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
	double sum = 0.0;

	// Sum out contribution of the errors at the nodes we feed
	for (unsigned n = 0; n < nextLayer.size() - 1; n++) {
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}
	return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer) 
{
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal)
{
	double delta = targetVal - m_outputVal;
	m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x) 
{
	// tanh - output range [-1.0 .. 1.0]
	return tanh(x);
}

double Neuron::transferFunctionDerivative(double th_x) 
{
	// tanh derivative 
	// d(tanh (x))/dx ~ 1 - tanh^2 (x)
	return 1.0 - th_x * th_x;
}

void Neuron::feedForward(const Layer &prevLayer)
{
	double sum = 0.0;

	// Sum the previous layer's output (which are our inputs)
	// Include the bias node from the previous layer.

	for (unsigned n = 0; n < prevLayer.size(); n++) {
		sum += prevLayer[n].getOutputVal() 
			* prevLayer[n].m_outputWeights[m_myIndex].weight;
	}

	m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
	for (unsigned c=0; c<numOutputs; c++) {
		Connection connection;
		connection.weight = randomWeight();
		connection.deltaWeight = 0.0;

		m_outputWeights.push_back(connection);
		cout << "Neuron connection " << c << " has a random weight of " << m_outputWeights.back().weight << endl;
	}

	m_myIndex = myIndex;
}