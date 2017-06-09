#include "stdafx.h"

std::ifstream m_learningFile;

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

void Net::printNetwork(const vector<unsigned> &topology, const double eta, const double alpha, const string learningFile) 
{
	unsigned numLayers = topology.size();
	ofstream m_learningDataFile;
	stringstream stream;
	string str;

	m_learningDataFile.open(learningFile);

	assert(m_learningDataFile.is_open() == true);
	if (!m_learningDataFile.is_open()) {
		cout << "cannot learn !" << endl; 
	}

	stream << string("topology: ");
	for (unsigned i = 0; i < topology.size()-1; i++) {
		stream << topology[i] << string(" ");
	}
	stream << topology[topology.size()-1];
	
	str = stream.str();

	cout << str << endl;
	m_learningDataFile << str << endl;

	cout << string("eta: ") << eta << endl;
	m_learningDataFile << string("eta: ") << eta << endl;

	cout << string("alpha: ") << alpha << endl;
	m_learningDataFile << string("alpha: ") << alpha << endl;

	for (unsigned layerNum = 0; layerNum < numLayers; layerNum++) {
		for (unsigned neuronNum = 0; neuronNum < m_layers[layerNum].size(); neuronNum++) {
			// print weight and delta weight for each Neuron in the layer
			unsigned numOutputs = (layerNum == topology.size()-1)? 0 : topology[layerNum+1];
			for (unsigned connection = 0; connection < numOutputs; connection++) {
				stream.str("");
				double weight = m_layers[layerNum][neuronNum].getNeuronWeights()[connection].weight;
				double deltaWeight = m_layers[layerNum][neuronNum].getNeuronWeights()[connection].deltaWeight;
				
				stream << "Layer: " << layerNum << " Neuron: " << neuronNum 
					<< " Connection: " << connection 
					<< " Weight: " << weight 
					<< " deltaWeight: " << deltaWeight;

				str = stream.str();
				cout << str << endl;
				m_learningDataFile << str << endl;

			} // loop for each neuron connection
		} // loop for each neuron in the layer
	} // loop for each layer in the net
	
	m_learningDataFile.close();
}

void Net::LearningData(const string filename)
{
	m_learningFile.open(filename);
}

bool Net::isEof(void) 
{
	return m_learningFile.eof();
}

unsigned Net::getTopology(vector<unsigned> &topology)
{
	string line;
	string label;

	getline(m_learningFile, line);
	stringstream ss(line);
	ss >> label;

	if (this->isEof() || label.compare("topology:") != 0) {
		abort();	
	}

	while (!ss.eof()) {			
		unsigned n;
		ss >> n;
		topology.push_back(n);
	}

	return topology.size();
}

void Net::getNetworkWeights(unsigned layerNum, unsigned neuronNum, unsigned connectionNum, Connection &connection)
{
	string line;
	string label;

	while (!this->isEof()) {
		getline(m_learningFile, line);
		stringstream ss(line);
		ss >> label;

		if (label.compare("Layer:") == 0) {
			unsigned layer;
			ss >> layer;
			if (layerNum == layer) {
				ss >> label;
				if (label.compare("Neuron:") == 0) {
					unsigned neuron;
					ss >> neuron;
					if (neuronNum == neuron) {
						ss >> label;
						if (label.compare("Connection:") == 0) {
							unsigned connect;
							ss >> connect;
							if (connectionNum == connect) {
								ss >> label;
								if (label.compare("Weight:") == 0) {
									double weight;
									ss >> weight;
									connection.weight = weight;

									ss >> label;
									if (label.compare("deltaWeight:") == 0) {
										double delatWeight;
										ss >> delatWeight;
										connection.deltaWeight = delatWeight;
										return;
									}
									else {
										abort();
									}
								}
								else {
									abort();
								}
							}
						}
						else {
							abort();
						}
					}
				}
				else {
					abort();
				}
			}
		}
		else {
			abort();
		}
	}
}

void Net::setNetworkWeights(const vector<unsigned> &topology)
{
	unsigned numLayers = topology.size();
	string line;
	string label;
	
	getline(m_learningFile, line);
	stringstream ss(line);

	ss >> label;
	if (label.compare("eta:") == 0) {
		double eta;
		ss >> eta;

		Neuron::eta = eta;
	} else {
		abort();
	}
	
	ss.clear();
	ss.str("");
	getline(m_learningFile, line);
	ss << line;

	ss >> label;
	if (label.compare("alpha:") == 0) {
		double alpha;
		ss >> alpha;

		Neuron::alpha = alpha;
	} else {
		abort();
	}

	for (unsigned layerNum = 0; layerNum < numLayers; layerNum++) {
		for (unsigned neuronNum = 0; neuronNum < m_layers[layerNum].size(); neuronNum++) {
			// print weight and delta weight for each Neuron in the layer
			unsigned numOutputs = (layerNum == topology.size()-1)? 0 : topology[layerNum+1];
			for (unsigned connection = 0; connection < numOutputs; connection++) {
				Connection neuronWeight;
				getNetworkWeights(layerNum, neuronNum, connection, neuronWeight);
				m_layers[layerNum][neuronNum].setNeuronWeights(connection,neuronWeight);

				stringstream stream;
				string str;

				stream << "Layer: " << layerNum << " Neuron: " << neuronNum 
					<< " Connection: " << connection 
					<< " Weight: " << neuronWeight.weight 
					<< " deltaWeight: " << neuronWeight.deltaWeight;

				str = stream.str();
				cout << str << endl;

			} // loop for each neuron connection
		} // loop for each neuron in the layer

		// force the bias node's output value to 1.0. It's the last neuron created above
		m_layers.back().back().setOutputVal(1.0);
	} // loop for each layer in the net	
}
/*
Net::Net(vector<unsigned> &topology, const string filename)
{
	Net::LearningData(filename);
	//vector<unsigned> topology;
	Net::getTopology(topology);

	Net myNet = Net::Net(topology);

	myNet.setNetworkWeights(topology);
}
*/
Net::Net(vector<unsigned> &topology, const string networkWeightsfilename, bool execute)
{
	if (execute) {
		LearningData(networkWeightsfilename);
		getTopology(topology);
	}

	unsigned numLayers = topology.size();
	// cout << "Num of Layers: " << numLayers << endl;

	for (unsigned layerNum = 0; layerNum < numLayers; layerNum++) {
		m_layers.push_back(Layer());
		unsigned numOutputs = (layerNum == topology.size()-1)? 0 : topology[layerNum+1];
		
		// we have made a new layer, now fill it with neurons, and
		// add a bias neuron to the layer
		// cout << "Made a Layer! Layer num:" << layerNum << endl;
		// cout << "## This Layer has " << topology[layerNum] << " Neurons" << endl;

		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++) {
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
			// cout << "Made a Neuron!" << endl;
		}
		// force the bias node's output value to 1.0. It's the last neuron created above
		m_layers.back().back().setOutputVal(1.0);
	}

	if (execute) {
		setNetworkWeights(topology);
	}
}