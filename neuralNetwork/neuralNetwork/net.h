#ifndef NET_H
#define NET_H

using namespace std;

class Neuron;

// ************* Class Net ***************
class Net
{
public:
	Net(const vector<unsigned> &topology);
	void feedForward(const std::vector<double> &inputVals);
	void backProp(const std::vector<double> &targetVals);
	void getResuls(std::vector<double> &resultVals) const;
	double getRecentAverageError(void) const {return m_recentAverageError;};

private:
	vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
	double m_error;
	double m_recentAverageError;
	double m_recentAverageSmoothingFactor;
};

#endif
