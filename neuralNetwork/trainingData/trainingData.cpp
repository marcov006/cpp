// trainingData.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <cstdlib>


using namespace std;

int NUM_OF_DATA = 6000;

int main(int argc, char* argv[])
{
	ofstream myfile;

	if ((argc > 1) && !((*argv[1] == 'y') || (*argv[1] == 'n'))) {
		myfile.open(argv[1]);
	} else {
		myfile.open("C:\\temp\\trainingData.txt");
	}

	unsigned inputLayer = 2;
	unsigned hiddenLayer = 4;
	unsigned outputLayer = 1;

	if (argc > 1) {
		for (int i = 1; i < argc; i++) {
			if (*argv[i] == 'y') {
				cout << "Type number of Neurons of input Layer:";
				cin >> inputLayer;
				cout << "Type number of Neurons of hidden Layer:";
				cin >> hiddenLayer;
				cout << "Type number of Neurons of output Layer:";
				cin >> outputLayer;
			}
		}
	}

	if (myfile.is_open()) {
		stringstream stream;
		string str;

		// Random training sets for XOR -- two inputs and one output
		stream << string("topology: ") << inputLayer << string(" ") << hiddenLayer << string(" ") << outputLayer;
		str = stream.str();

		cout << str << endl;
		myfile << str << endl;

		for (int i = NUM_OF_DATA; i >= 0; i--) {
			int n1 = (int)(2.0 * rand() / double(RAND_MAX));
			int n2 = (int)(2.0 * rand() / double(RAND_MAX));
			int t = n1 | n2; // should be 0 or 1
			stream.str("");
			stream << string("in: ") << n1 << string(".0 ") << n2 << string(".0");
			str = stream.str();

			cout << str << endl;
			myfile << str << endl;
			
			stream.str("");
			stream << string("out: ") << t << string(".0");
			str = stream.str();

			cout << str << endl;
			myfile << str << endl;
		}
	}
}


