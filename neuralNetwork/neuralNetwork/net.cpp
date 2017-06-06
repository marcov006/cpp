#include "stdafx.h"

void Net::getResuls(std::vector<double> &resultVals) const 
{
	resultVals.clear();

	for (unsigned n = 0; n < m_layers.back().size() - 1; n++) {
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}

void Net::backProp(const std::vector<double> &targetVals)
{
	// Calculate overall net error (RMS of output neuron errors)
	Layer &outputLayer = m_layers.back();
	m_error = 0.0;

	for (unsigned n = 0; n < outputLayer.size() - 1; n++) {
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += delta * delta;
	}

	m_error /= outputLayer.size(); // average the error squared
	m_error = sqrt(m_error); // RMS

	// Implement a recent average measurement:
	// m_recentAverageError = 
	//	(m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
	//	/ (m_recentAverageSmoothingFactor + 1.0);
	m_recentAverageError = m_error;

	// Calculate output layer gradients
	for (unsigned n = 0; n < outputLayer.size() - 1; n++) {
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	// Calculate gradients on hidden layers
	for (unsigned layerNum =  m_layers.size() - 2; layerNum > 0; layerNum--) {
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum+1];

		for (unsigned n = 0; n < hiddenLayer.size(); n++) {
			hiddenLayer[n].calcHiddenGradients(nextLayer);	
		}
	}

	// For all layers from outputs to first hidden layer
	// update connection weights
	for (unsigned layerNum =  m_layers.size() - 1; layerNum > 0; layerNum--) {
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];

		for (unsigned n = 0; n < layer.size() - 1; n++) {
			layer[n].updateInputWeights(prevLayer);	
		}	
	}
}

// Class to latch the input values into the first stage of input neurons
// This is to feed the NN
// Then forward the values to the following NN layers
void Net::feedForward(const std::vector<double> &inputVals)
{
	// assert if the num of input value is different from the num of neuron in 1st layer.
	assert(inputVals.size() == m_layers[0].size() - 1); 

	// Assign / latch the input values into the input neurons
	for (unsigned i=0; i<inputVals.size(); i++) {
		m_layers[0][i].setOutputVal(inputVals[i]);
	}

	// Forward propagate
	// Each Layer N, has its neurons inputs values equals to Layer-1 neurons values x weight of the connection
	for (unsigned layerNum = 1; layerNum < m_layers.size(); layerNum++) {
		Layer &prevLayer = m_layers[layerNum-1];
		for (unsigned neuronsNum=0; neuronsNum<(m_layers[layerNum].size()-1); neuronsNum++) {
			m_layers[layerNum][neuronsNum].feedForward(prevLayer);
		}
	}
}

Net::Net(const vector<unsigned> &topology)
{
	unsigned numLayers = topology.size();
	cout << "Num of Layers: " << numLayers << endl;

	for (unsigned layerNum = 0; layerNum < numLayers; layerNum++) {
		m_layers.push_back(Layer());
		unsigned numOutputs = (layerNum == topology.size()-1)? 0 : topology[layerNum+1];
		
		// we have made a new layer, now fill it with neurons, and
		// add a bias neuron to the layer
		cout << "Made a Layer! Layer num:" << layerNum << endl;
		cout << "## This Layer has " << topology[layerNum] << " Neurons" << endl;

		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++) {
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
			cout << "Made a Neuron!" << endl;
		}
		// force the bias node's output value to 1.0. It's the last neuron created above
		m_layers.back().back().setOutputVal(1.0);
	}
}