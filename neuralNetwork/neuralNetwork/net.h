#ifndef NET_H
#define NET_H

using namespace std;

class Neuron;

// ************* Class Net ***************
class Net
{
public:
	//Net(const vector<unsigned> &topology);
	Net(vector<unsigned> &topology, const string networkWeightsfilename, bool execute);
	Net(vector<unsigned> &topology, const string filename);
	
	void printNetwork(const vector<unsigned> &topology, const double eta, const double alpha, const string learningFile);
	void feedForward(const std::vector<double> &inputVals);
	void backProp(const std::vector<double> &targetVals);
	void getResuls(std::vector<double> &resultVals) const;
	double getRecentAverageError(void) const {return m_recentAverageError;};

private:
	vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
	double m_error;
	double m_recentAverageError;
	double m_recentAverageSmoothingFactor;
		
	void setNetworkWeights(const vector<unsigned> &topology);
	void getNetworkWeights(unsigned layerNum, unsigned neuronNum, unsigned connectionNum, Connection &connection);
	unsigned getTopology(vector<unsigned> &topology);
	bool isEof(void);
	void LearningData(const string filename);
};

#endif
