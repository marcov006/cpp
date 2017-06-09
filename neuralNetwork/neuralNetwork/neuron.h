#ifndef NEURON_H
#define NEURON_H

using namespace std;

// ************* Struct, typedef, and declarations ***************
struct Connection {
	double weight;
	double deltaWeight;
};

// ************* Class Neuron ***************
class Neuron
{
public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	void setOutputVal(double val) { m_outputVal = val; }
	double getOutputVal(void) const { return m_outputVal; }
	vector<Connection> getNeuronWeights(void) { return m_outputWeights; }
	void setNeuronWeights(unsigned connectionNum, Connection neuronWeights) { m_outputWeights[connectionNum]=neuronWeights; }
	void feedForward(const Layer &prevLayer);
	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);
	static double eta; // [0.0..1.0] overall net training rate
	static double alpha; // [0.0..n] mulitplier of last weight change (momentum)
	
private:
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	static double randomWeight(void) { return (rand() / double(RAND_MAX)); }
	double sumDOW(const Layer &nextLayer) const;
	double m_outputVal;
	vector<Connection> m_outputWeights;	
	unsigned m_myIndex;
	double m_gradient;
};

#endif