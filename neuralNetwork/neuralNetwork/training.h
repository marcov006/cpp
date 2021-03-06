#ifndef TRAINING_H
#define TRAINING_H

using namespace std;

// Class to read training data from a file
class TrainingData
{
public:
	TrainingData(const string filename);
	bool isEof(void);
	unsigned getTopology(vector<unsigned> &topology);

	// returns the number of input values read from the file:
	unsigned getNextInputs(vector<double> &inputVals);
	unsigned getTargetOutputs(vector<double> &targetOutputVals);
	void showVectorVals(string label, vector<double> &v, bool printOut);

private:
	ifstream m_trainingDataFile;
};

#endif