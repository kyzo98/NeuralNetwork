#pragma once
#include<vector>
#include<assert.h>
#include<math.h>
#include<iostream>
#include<fstream>
#include<sstream>

#define Layer std::vector<Neuron>

// ***************** Class TrainingData *****************
class TrainingData {
public:
	TrainingData(const std::string filename);
	bool IsEof();
	void GetTopology(std::vector<unsigned> &topology);

	//Returns the number of input values read from the file
	unsigned GetNextInputs(std::vector<double> &inputVals);
	unsigned GetTargetOutputs(std::vector<double> &targetOutputVals);

private:
	std::ifstream trainingDataFile;
};

// ***************** Struct Connection *****************
struct Connection {
	double weight;
	double deltaWeight;
};

// ***************** Class Neuron *****************
class Neuron {
public:
	Neuron(unsigned numOutputs, unsigned thisIndex);

	void FeedForward(const Layer &prevLayer);
	void SetOutputVal(double val);
	double GetOutputVal() const;
	void SetData(std::vector<double> weights);
	std::vector<Connection> GetData() const;
	void CalcOutputGradients(double targetVal);
	void CalcHiddenGradients(const Layer &nextLayer);
	void UpdateInputWeights(Layer &prevLayer);
	std::vector<Connection> outputWeights;

private:
	static double eta;	//[0.0..1.0] overall net training rate
	static double alpha;//[0.0..n] multiplier of last weight change (momentum)
	double outputVal;
	double gradient;
	unsigned myIndex;

	static double TransferFunction(double x);
	static double TransferFunctionDerivative(double x);
	static double RandomWeight();
	double SumDOW(const Layer &nextLayer) const;
};

// ***************** Class Net *****************
class Net {
private:
	unsigned numLayers;
	double error;
	double recentAverageError;
	double recentAverageSmoothingFactor;

	std::vector<Layer> layers;	//layers[layerNum][neuronNum]

	void SetData(std::vector<double> data);

public:
	Net(const std::vector<unsigned> &topology);

	void FeedForward(const std::vector<double> &inputVals);
	void BackProp(const std::vector<double> &targetVals);
	void GetResults(std::vector<double> &resultVals) const;
	void SaveNet() const;
	void LoadNet();
	void PrintConnections() const;
	double GetRecentAverageError() const;
};

