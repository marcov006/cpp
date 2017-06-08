// https://vimeo.com/19569529
// http://www.millermattson.com/dave/?p=54

// neuralNetwork.cpp : Defines the entry point for the console application.
#include "stdafx.h"
#include "training.h"
#include "neuron.h"
#include "net.h"

int main(int argc, char *argv[])
{
	bool printOut = false;
		
	string trainingFile="";
	string learningFile="";
	
	trainingFile = "C:\\temp\\trainingData.txt";
	learningFile = "C:\\temp\\NeuralNetworkTrainingSequence.txt";

	if (argc > 1) {
		for (int i = 0; i < argc; i++) {
			cout << "argc: " << i << endl;
			cout << "argv: " << argv[i] << endl;
		}
		for (int i = 1; i < argc-1; i++) {
			string arg_s = argv[i];
			if (arg_s.compare("-f1") == 0) {
				trainingFile = argv[i+1];
				cout << "trainingFile: " << trainingFile << endl;
			}
			else if (arg_s.compare("-f2") == 0) {
				learningFile = argv[i+1];
				cout << "learningFile: " << learningFile << endl;
			}
			else if (arg_s.compare("-eta") == 0) {
				Neuron::eta = (double)atof(argv[i+1]);
				cout << "eta: " << Neuron::eta << endl;
			}
			else if (arg_s.compare("-alpha") == 0) {
				Neuron::alpha = (double)atof(argv[i+1]);
				cout << "alpha: " << Neuron::alpha << endl;
			}
			else {
				// cout << "option : " << arg_s << " does not exist" << endl;
			}
		}
	} else {
		trainingFile = "C:\\temp\\trainingData.txt";
		learningFile = "C:\\temp\\NeuralNetworkTrainingSequence.txt";
	}

	cout << " trainingFile = " << trainingFile << endl;
	cout << "eta: " << Neuron::eta << endl;
	cout << "alpha: " << Neuron::alpha << endl;

	for (double i = 0.0; i < 21.0; i++) {
		for (double j = 0.0; j < 21.0; j++) {
			TrainingData trainData(trainingFile, learningFile);

			cout << "## Learning Loop count : " << (i*20)+j << endl;

			Neuron::eta = i*0.05;
			Neuron::alpha = j*0.05;

			cout << ":eta:" << Neuron::eta ;
			cout << ":alpha:" << Neuron::alpha;

			// e.g.,  {3, 2, 1}
			vector<unsigned> topology;
			trainData.getTopology(topology);
			Net myNet(topology);

			std::vector<double> inputVals, targetVals, resultVals;
			int trainingPass = 0;

			bool recordTrainingPass = false;
			unsigned learnEndIteration = 0;

			while (!trainData.isEof()) {
				++trainingPass;
		
				double recentAverageError = 0.0;

				if (printOut) 
					cout << endl << "Pass " << trainingPass <<  endl;

				// Get new input data and feed it forward;
				if (trainData.getNextInputs(inputVals) != topology[0]) {
					break;
				}
				trainData.showVectorVals(": Inputs:", inputVals, printOut);
				myNet.feedForward(inputVals);

				// Collect the net's actual results:
				myNet.getResuls(resultVals);
				trainData.showVectorVals(": Outputs:", resultVals, printOut);

				// Train the net what the output should have been:
				trainData.getTargetOutputs(targetVals);
				trainData.showVectorVals(": Targets:", targetVals, printOut);
				assert(targetVals.size() == topology.back());

				myNet.backProp(targetVals);

				recentAverageError = myNet.getRecentAverageError();

				// report how well the training is working, averaged over recent samples:
				if (printOut)
					cout << "Net recent average error: "
						<< recentAverageError << endl;

				if (recordTrainingPass)
					learnEndIteration = trainingPass;

				if (recentAverageError >= 0.05)
					recordTrainingPass = true;
				else
					recordTrainingPass = false;

				if (printOut)
					cout << "Record training pass: " 
						<< recordTrainingPass << endl;
			} // while training data
	
			cout << ":Learning was completed on Training Pass:" << learnEndIteration << endl;
			//cout << "Done. " << endl;

		} // loop alpha
	} // loop eta

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