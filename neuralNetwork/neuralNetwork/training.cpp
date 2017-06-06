#include "stdafx.h"

bool TrainingData::isEof()
{
	return m_trainingDataFile.eof(); 
}

unsigned TrainingData::getTopology(vector<unsigned> &topology)
{
	string line;
	string label;

	getline(m_trainingDataFile, line);
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

unsigned TrainingData::getNextInputs(vector<double> &inputVals)
{
	inputVals.clear();

	string line;
	string label;

	getline(m_trainingDataFile, line);
	stringstream ss(line);
	ss >> label;

	if (label.compare("in:") == 0) {
		double oneValue;
		while (!ss.eof()) {
			ss >> oneValue;
			inputVals.push_back(oneValue);
		}		
	}

	return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals)
{
	targetOutputVals.clear();

	string line;
	string label;

	getline(m_trainingDataFile, line);
	stringstream ss(line);
	ss >> label;

	if (label.compare("out:") == 0) {
		double oneValue;
		while (!ss.eof()) {
			ss >> oneValue;
			targetOutputVals.push_back(oneValue);
		}		
	}

	return targetOutputVals.size();
}

TrainingData::TrainingData(const string filename, const string filename2)
{
	m_trainingDataFile.open(filename);
	m_learningDataFile.open(filename2);

	if (m_trainingDataFile.is_open())
		cout << "ok. " << endl;
	else
		cout << "pas bon ca." <<endl;

	if (m_learningDataFile.is_open())
		cout << "ok. " << endl;
	else
		cout << "pas bon ca." <<endl;
}

void TrainingData::showVectorVals(string label, vector<double> &v)
{
	cout << label << " ";
	m_learningDataFile << label << " " << endl;

	for (unsigned i = 0; i < v.size(); i++) {
		cout << v[i] << " ";
		m_learningDataFile << v[i] << " ";
	}
	cout << endl;
	m_learningDataFile << endl;
}