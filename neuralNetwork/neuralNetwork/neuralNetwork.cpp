// https://vimeo.com/19569529
// http://www.millermattson.com/dave/?p=54

// neuralNetwork.cpp : Defines the entry point for the console application.
#include "stdafx.h"
#include "training.h"
#include "neuron.h"
#include "net.h"

int main(int argc, char argv[])
{
	string trainingFile="";
	string learningFile="";

	if (argc > 1) {
		trainingFile = argv[1];
		learningFile = argv[2];
	} else {
		trainingFile = "C:\\temp\\trainingData.txt";
		learningFile = "C:\\temp\\NeuralNetworkTrainingSequence.txt";
	}

	TrainingData trainData(trainingFile, learningFile);

	// e.g.,  {3, 2, 1}
	vector<unsigned> topology;
	trainData.getTopology(topology);
	Net myNet(topology);

	std::vector<double> inputVals, targetVals, resultVals;
	int trainingPass = 0;

	while (!trainData.isEof()) {
		++trainingPass;
		cout << endl << "Pass " << trainingPass <<  endl;

		// Get new input data and feed it forward;
		if (trainData.getNextInputs(inputVals) != topology[0]) {
			break;
		}
		trainData.showVectorVals(": Inputs:", inputVals);
		myNet.feedForward(inputVals);

		// Collect the net's actual results:
		myNet.getResuls(resultVals);
		trainData.showVectorVals(": Outputs:", resultVals);

		// Train the net what the output should have been:
		trainData.getTargetOutputs(targetVals);
		trainData.showVectorVals(": Targets:", targetVals);
		assert(targetVals.size() == topology.back());

		myNet.backProp(targetVals);

		// report how well the training is working, averaged over recent samples:
		cout << "Net recent average error: "
			<< myNet.getRecentAverageError() << endl;
	}

	cout << "Done. " << endl;
}

/*
int main()
{
	// e.g.,  {3, 2, 1}
	std::vector<unsigned> topology;
	topology.push_back(3);
	topology.push_back(2);
	topology.push_back(1);
	Net myNet(topology);

	std::vector<double> inputVals;
	myNet.feedForward(inputVals);

	std::vector<double> targetVals;
	myNet.backProp(targetVals);

	std::vector<double> resultVals;
	myNet.getResuls(resultVals);
}
*/

/*
int _tmain(int argc, _TCHAR* argv[])
{
	return 0;
}
*/