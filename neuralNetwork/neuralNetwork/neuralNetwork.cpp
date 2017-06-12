// https://vimeo.com/19569529
// http://www.millermattson.com/dave/?p=54

// neuralNetwork.cpp : Defines the entry point for the console application.
#include "stdafx.h"
#include "training.h"
#include "neuron.h"
#include "net.h"

using namespace std;

int main(int argc, char *argv[])
{
	bool printOut = false;
	bool singleShot = false;

	string trainingFile="";
	string learningFile="";
	string nNmode = "training";

	trainingFile = "C:\\temp\\trainingData.txt";
	learningFile = "C:\\temp\\NeuralNetworkTrainingSequence.txt";
	
	Neuron::eta = 0.05;
	Neuron::alpha = 0.8;

	if (argc > 1) {
		for (int i = 0; i < argc; i++) {
			std::cout << "argc: " << i << endl;
			cout << "argv: " << argv[i] << endl;

			string arg_s = argv[i]; 
			if ((arg_s.compare("-h") == 0) || (arg_s.compare("-help") == 0)) {
				cout << "Help Menu: " << endl;
				cout << "\t# -f1 : specify the training file to learn/execute the Neural Network [default: C:\\temp\\trainingData.txt]" << endl;
				cout << "\t# -f2 : specify the learning file to store the best weights [default: C:\\temp\\NeuralNetworkTrainingSequence.txt]" << endl;
				cout << "\t# -eta : fix the ETA value, usefull with oneshot=y [default Neuron::eta=" << Neuron::eta << "]" << endl;
				cout << "\t# -alpha : fix the ALPHA value, usefull with oneshot=y [default Neuron::alpha=" << Neuron::alpha << "]" << endl;
				cout << "\t# -mode : train or execute the Neural Network. Both needs training and learning file. [training (default) /execute]" << endl;
				cout << "\t# -print : verbose or not [y/n(default)] " << endl;
				cout << "\t# -oneshot : train the network for a single set of eta and alpha or not [y/n(default)] " << endl;
				return 0;
			}
		}
		for (int i = 1; i < argc-1; i+=2) {
			string arg_s = argv[i];

			cout << "arg_s: " << arg_s << endl;

			if (arg_s.compare("-f1") == 0) {
				trainingFile = argv[i+1];
				cout << "trainingFile: " << trainingFile << endl;
			}
			else if (arg_s.compare("-f2") == 0) {
				learningFile = (string)argv[i+1];
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
			else if (arg_s.compare("-mode") == 0) {
				nNmode = (string)argv[i+1];
				cout << "nNmode: " << nNmode << endl;

				if ((nNmode.compare("training") != 0) && (nNmode.compare("execute") != 0)) {
					cout << "enter a valid value for -mode (training or execute) : " << endl;					
					cin >> nNmode;
					if ((nNmode.compare("training") != 0) && (nNmode.compare("execute") != 0)) {
						cout << "t'es vraiment con, je peux plus rien pour toi ! Hasta luego" << endl;
						return 0;
					}
				}
			}
			else if (arg_s.compare("-print") == 0) {
				string printConsole = (string)argv[i+1];
				cout << "print: " << printConsole << endl;

				if ((printConsole.compare("y") != 0) && (printConsole.compare("n") != 0)) {
					cout << "enter a valid value for -print (y or n) : " << endl;					
					cin >> printConsole;
					if ((printConsole.compare("y") != 0) && (printConsole.compare("n") != 0)) {
						cout << "t'es vraiment con, je peux plus rien pour toi ! Hasta luego" << endl;
						return 0;
					}
				}

				if (printConsole.compare("y") == 0)
					printOut = true;
				else
					printOut = false;

			}
			else if (arg_s.compare("-oneshot") == 0) {
				string oneshot = (string)argv[i+1];
				cout << "oneshot: " << oneshot << endl;

				if ((oneshot.compare("y") != 0) && (oneshot.compare("n") != 0)) {
					cout << "enter a valid value for -oneshot (y or n) : " << endl;					
					cin >> oneshot;
					if ((oneshot.compare("y") != 0) && (oneshot.compare("n") != 0)) {
						cout << "t'es vraiment con, je peux plus rien pour toi ! Hasta luego" << endl;
						return 0;
					}
				}
				if (oneshot.compare("y") == 0)
					singleShot = true;
				else
					singleShot = false;
			}
			else {
				cout << "option : " << arg_s << " does not exist" << endl;
				cout << "Help Menu: " << endl;
				cout << "\t# -f1 : specify the training file to learn/execute the Neural Network [default: C:\\temp\\trainingData.txt]" << endl;
				cout << "\t# -f2 : specify the learning file to store the best weights [default: C:\\temp\\NeuralNetworkTrainingSequence.txt]" << endl;
				cout << "\t# -eta : fix the ETA value, usefull with oneshot=y [default Neuron::eta=" << Neuron::eta << "]" << endl;
				cout << "\t# -alpha : fix the ALPHA value, usefull with oneshot=y [default Neuron::alpha=" << Neuron::alpha << "]" << endl;
				cout << "\t# -mode : train or execute the Neural Network. Both needs training and learning file. [training (default) /execute]" << endl;
				cout << "\t# -print : verbose or not [y/n(default)] " << endl;
				cout << "\t# -oneshot : train the network for a single set of eta and alpha or not [y/n(default)] " << endl;
				return 0;
			}
		}
	}

	cout << "mode: " << nNmode << endl;
	cout << "trainingFile = " << trainingFile << endl;
	cout << "learningFile = " << learningFile << endl;
	
	if (nNmode.compare("training") == 0) {
		if (!singleShot) {
			unsigned currentLearnIteration = UINT_MAX;
			double currentETA = 0.0;
			double currentALPHA = 0.0;

			unsigned bestAlphaIt = UINT_MAX;
			double bestAlpha_ETA = 0.0;
			double bestAlpha_ALPHA = 0.0;

			unsigned bestLearnIteration = UINT_MAX;
			double bestETA = 0.0;
			double bestALPHA = 0.0;

			TrainingData tData(trainingFile);
			vector<unsigned> topo;
			tData.getTopology(topo);

			Net myBestAlphaLearntNet(topo, "nofile", false);
			Net myBestLearntNet(topo, "nofile", false);

			cout << "Learning .";

			for (double i = 0.0; i < 21.0; i++) { // eta loop
				cout << ".";
				for (double j = 0.0; j < 21.0; j++) { // alpha loop
					TrainingData trainData(trainingFile);			

					Neuron::eta = i*0.05;
					Neuron::alpha = j*0.05;

					currentETA = Neuron::eta;
					currentALPHA = Neuron::alpha;

					if (printOut) {
						cout << "## Learning Loop count : " << (i*20)+j << endl;
						cout << ":eta:" << Neuron::eta ;
						cout << ":alpha:" << Neuron::alpha;
					}

					vector<unsigned> topology;
					std::vector<double> inputVals, targetVals, resultVals;
					int trainingPass = 0;

					trainData.getTopology(topology);

					srand(time_t(0));
					Net myNet(topology, "nofile", false);				

					bool recordTrainingPass = false;

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

						if (recordTrainingPass) {
							currentLearnIteration = trainingPass;							
						}

						if (recentAverageError >= 0.05)
							recordTrainingPass = true;
						else
							recordTrainingPass = false;
						
						if (printOut)
							cout << "Record training pass: " 
							<< recordTrainingPass << endl;

					} // while training data

					if (printOut)
						cout << ":Learning was completed on Training Pass:" << currentLearnIteration << endl;

					if (currentLearnIteration < bestAlphaIt) {
						bestAlphaIt = currentLearnIteration;
						bestAlpha_ETA = currentETA;
						bestAlpha_ALPHA = currentALPHA;
						myBestAlphaLearntNet = myNet;

						if (printOut) {
							cout << "Update bestAlphaIt ..." << endl;
							cout << "- bestAlphaIt: " << bestAlphaIt 
								<< "- bestAlpha_ETA: " << bestAlpha_ETA 
								<< "- bestAlpha_ALPHA: " << bestAlpha_ALPHA 
								<< endl; 
						}
					}
				} // loop alpha

				if (bestAlphaIt < bestLearnIteration) {
					bestLearnIteration = bestAlphaIt;
					bestETA = bestAlpha_ETA;
					bestALPHA = bestAlpha_ALPHA;
					myBestLearntNet = myBestAlphaLearntNet;

					if (printOut) {
						cout << "Update bestLearnIteration ..." << endl;
						cout << "- bestLearnIteration: " << bestLearnIteration 
							<< "- bestETA: " << bestETA 
							<< "- bestALPHA: " << bestALPHA 
							<< endl;
					}
				}

				bestAlphaIt = UINT_MAX;
			} // loop eta

			cout << endl;
			cout << "Learning Done" << endl << endl;

			cout << "- bestLearnIteration: " << bestLearnIteration 
				<< "- bestETA: " << bestETA 
				<< "- bestALPHA: " << bestALPHA 
				<< endl; 

			myBestLearntNet.printNetwork(topo, bestETA, bestALPHA, learningFile);
		}

		if (singleShot) {

			if (printOut) {
				cout << ":eta:" << Neuron::eta ;
				cout << ":alpha:" << Neuron::alpha;
			}

			TrainingData trainData(trainingFile);
			vector<unsigned> topology;
			std::vector<double> inputVals, targetVals, resultVals;
			int trainingPass = 0;
			unsigned currentLearnIteration = 0;

			trainData.getTopology(topology);
			Net myNet(topology, "nofile", false);

			bool recordTrainingPass = false;

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

				if (recordTrainingPass) {
					currentLearnIteration = trainingPass;
				}

				if (recentAverageError >= 0.05)
					recordTrainingPass = true;
				else
					recordTrainingPass = false;

				if (printOut)
					cout << "Record training pass: " 
					<< recordTrainingPass << endl;
			}
			cout << ":Learning was completed on Training Pass:" << currentLearnIteration << endl;

			myNet.printNetwork(topology, Neuron::eta,Neuron::alpha,learningFile);
		}
	} // if(training)
	else if (nNmode.compare("execute") == 0) {
	
		vector<unsigned> topology;
		std::vector<double> inputVals, targetVals, resultVals;
		int trainingPass = 0;

		Net myNet(topology, learningFile, true);

		TrainingData trainData(trainingFile);
		vector<unsigned> trainTopology;
		trainData.getTopology(trainTopology);

		assert(topology == trainTopology);
		
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

			// Get Target output val
			trainData.getTargetOutputs(targetVals);
			trainData.showVectorVals(": Targets:", targetVals, printOut);
			assert(targetVals.size() == topology.back());

			double delta = targetVals[targetVals.size()-1] - resultVals[targetVals.size()-1];

			// report how well the training is working, averaged over recent samples:
			if (printOut)
				cout << "Net Absolute error: "
				<< delta << endl;

			if (abs(delta) > 0.05) {
				cout << "Net Absolute error ABOVE the threshold fixed during the learning sequence : " << delta << endl;
				return 0;
			}
		}
	} // if (execute)
	return 0;
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