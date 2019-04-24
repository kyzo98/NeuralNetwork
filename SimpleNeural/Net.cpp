#include "Net.h"

// ***************** Class Neuron *****************
Neuron::Neuron(unsigned numOutputs, unsigned thisIndex)
{
	for (unsigned c = 0; c < numOutputs; c++)
	{
		outputWeights.push_back(Connection());
		outputWeights.back().weight = RandomWeight();
	}
	myIndex = thisIndex;
}

void Neuron::FeedForward(const Layer &prevLayer)
{
	double sum = 0.0;

	for (unsigned n = 0; n < prevLayer.size(); n++)
	{
		sum += prevLayer[n].GetOutputVal() * prevLayer[n].outputWeights[myIndex].weight;
	}

	outputVal = Neuron::TransferFunction(sum);
}

void Neuron::SetOutputVal(double val)
{
	outputVal = val;
}

double Neuron::GetOutputVal() const
{
	return outputVal;
}

void Neuron::SetData(std::vector<double> weights)
{
	for (unsigned i = 0; i < outputWeights.size(); i++)
	{
		outputWeights[i].weight = weights[i];
	}
}

std::vector<Connection> Neuron::GetData() const
{
	return outputWeights;
}

void Neuron::CalcOutputGradients(double targetVal)
{
	double delta = targetVal - outputVal;
	gradient = delta * Neuron::TransferFunctionDerivative(outputVal);
}

void Neuron::CalcHiddenGradients(const Layer & nextLayer)
{
	double dow = SumDOW(nextLayer);
	gradient = dow * Neuron::TransferFunctionDerivative(outputVal);
}

void Neuron::UpdateInputWeights(Layer & prevLayer)
{
	//The weights to be updated are in the Connection container
	//In the neourons in the preceding layer
	for (unsigned n = 0; n < prevLayer.size(); n++)
	{
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.outputWeights[myIndex].deltaWeight;

		double newDeltaWeight =
			//Individual input, magnified by the gradient and train rate:
			eta
			* neuron.GetOutputVal()
			* gradient
			//Also add momentum = a fraction of the previous delta weight
			+ alpha
			* oldDeltaWeight;
		neuron.outputWeights[myIndex].deltaWeight = newDeltaWeight;
		neuron.outputWeights[myIndex].weight += newDeltaWeight;
	}
}

double Neuron::TransferFunction(double x)
{
	// tanh - output range [-1.0, 1.0]
	return tanh(x);
}

double Neuron::TransferFunctionDerivative(double x)
{
	// tanh derivative
	return 1.0 - x * x;
}

double Neuron::RandomWeight()
{
	return rand() / double(RAND_MAX);
}

double Neuron::SumDOW(const Layer & nextLayer) const
{
	double sum = 0.0;

	//Sum our contributions of the errors at the nodes we feed
	for (unsigned n = 0; n < nextLayer.size() - 1; n++)
	{
		sum += outputWeights[n].weight * nextLayer[n].gradient;
	}

	return sum;
}

double Neuron::eta = 0.15;	//Overall net learning rate, [0.0..1.0]
double Neuron::alpha = 0.5; //Momentum, multiplier of last deltaWeight, [0.0..n]


void Net::SetData(std::vector<double> data)
{
	for (unsigned layerNum = 0; layerNum < layers.size() - 1; layerNum++)
	{
		for (unsigned n = 0; n < layers[layerNum].size(); n++)
		{
			std::vector<double> aux;
			for (int i = 0; i < layers[layerNum][n].outputWeights.size(); i++)
			{
				aux.push_back(data.front());
				data.erase(data.begin());
			}
			layers[layerNum][n].SetData(aux);
		}
	}
}

Net::Net(const std::vector<unsigned>& topology)
{
	numLayers = topology.size();
	for (unsigned layerNum = 0; layerNum < numLayers; layerNum++)
	{
		layers.push_back(Layer());
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++) {	// = adds Bias neuron

			layers.back().push_back(Neuron(numOutputs, neuronNum));
		}

		//Force the bias node's output value to 1.0 it's the last neuron created above
		layers.back().back().SetOutputVal(1.0);
	}
}

void Net::FeedForward(const std::vector<double>& inputVals)
{
	assert(inputVals.size() == layers[0].size() - 1);

	//Assign the input values into the input neurons
	for (unsigned i = 0; i < inputVals.size(); i++)
	{
		layers[0][i].SetOutputVal(inputVals[i]);
	}

	//Forward propagate
	for (unsigned layerNum = 1; layerNum < layers.size(); layerNum++)
	{
		Layer &prevLayer = layers[layerNum - 1];
		for (unsigned n = 0; n < layers[layerNum].size() - 1; n++)
		{
			layers[layerNum][n].FeedForward(prevLayer);
		}
	}
}

void Net::BackProp(const std::vector<double>& targetVals)
{
	//Calculate overall net error (RMS of output neuron errors)
	Layer &outputLayer = layers.back();
	error = 0.0;

	for (unsigned n = 0; n < outputLayer.size() - 1; n++)
	{
		double delta = targetVals[n] - outputLayer[n].GetOutputVal();
		error += delta * delta;
	}
	error /= outputLayer.size() - 1;	//Get error average squared
	error = sqrt(error); //RMS

	//Implement a recent average measurament
	recentAverageError = (recentAverageError * recentAverageSmoothingFactor + error) / (recentAverageSmoothingFactor + 1.0);

	//Calculate output layer gradients
	for (unsigned n = 0; n < outputLayer.size() - 1 ;n++) 
	{
		outputLayer[n].CalcOutputGradients(targetVals[n]);
	}

	//Calculate gradients on hidden layers
	for (unsigned layerNum = layers.size() - 2; layerNum > 0; layerNum--)
	{
		Layer &hiddenLayer = layers[layerNum];
		Layer &nextLayer = layers[layerNum + 1];

		for (unsigned n = 0; n < hiddenLayer.size(); n++)
		{
			hiddenLayer[n].CalcHiddenGradients(nextLayer);
		}
	}

	//For all layers from outputs to first hidden layer
	//Update connection weights
	for (unsigned layerNum = layers.size() - 1; layerNum > 0; layerNum--) 
	{
		Layer &layer = layers[layerNum];
		Layer &prevLayer = layers[layerNum - 1];

		for (unsigned n = 0; n < layer.size() - 1; n++)
		{
			layer[n].UpdateInputWeights(prevLayer);
		}
	}
}

void Net::GetResults(std::vector<double>& resultVals) const
{
	resultVals.clear();

	for (unsigned n = 0; n < layers.back().size() - 1; n++)
	{
		resultVals.push_back(layers.back()[n].GetOutputVal());
	}
}

void Net::SaveNet() const
{
	std::ofstream saveDataFile;
	saveDataFile.open("savedData.txt");
	for (unsigned layerNum = 0; layerNum < layers.size() - 1; layerNum++)
	{
		for (unsigned n = 0; n < layers[layerNum].size(); n++)
		{
			for (int i = 0; i < layers[layerNum][n].GetData().size(); i++)
			{
				saveDataFile << layers[layerNum][n].GetData()[i].weight << " " << layers[layerNum][n].GetData()[i].deltaWeight << std::endl;
			}
		}
	}
	
	saveDataFile.close();
}

void Net::LoadNet()
{
	std::ifstream loadDataFile;
	loadDataFile.open("savedData.txt");

	std::vector<double> aux;
	while (!loadDataFile.eof()) {
		std::string line;
		std::getline(loadDataFile, line);
		std::stringstream ss(line);

		double oneValue;
		ss >> oneValue;
		aux.push_back(oneValue);
	}

	loadDataFile.close();

	SetData(aux);
}

void Net::PrintConnections() const
{
	for (unsigned layerNum = 0; layerNum < layers.size() - 1; layerNum++)
	{
		for (unsigned n = 0; n < layers[layerNum].size(); n++)
		{
			for (int i = 0; i < layers[layerNum][n].GetData().size(); i++)
			{
				std::cout << layers[layerNum][n].GetData()[i].weight << std::endl;
			}
		}
	}
}

double Net::GetRecentAverageError() const
{
	return recentAverageError;
}

// ***************** Class TrainingData *****************
TrainingData::TrainingData(const std::string filename)
{
	trainingDataFile.open(filename);
	if (!trainingDataFile.is_open()) std::cout << "Didnt found " << filename;
}

bool TrainingData::IsEof()
{
	return trainingDataFile.eof();
}

void TrainingData::GetTopology(std::vector<unsigned>& topology)
{
	std::string line;
	std::string label;

	std::getline(trainingDataFile, line);
	std::stringstream ss(line);
	ss >> label;
	if (this->IsEof() || label.compare("topology:") != 0)
		abort();

	while (!ss.eof()) {
		unsigned n;
		ss >> n;
		topology.push_back(n);
	}

	return;
}

unsigned TrainingData::GetNextInputs(std::vector<double>& inputVals)
{
	inputVals.clear();

	std::string line;
	std::getline(trainingDataFile, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;
	if (label.compare("in:") == 0) {
		double oneValue;
		while (ss >> oneValue) {
			inputVals.push_back(oneValue);
		}
	}

	return inputVals.size();
}

unsigned TrainingData::GetTargetOutputs(std::vector<double>& targetOutputVals)
{
	targetOutputVals.clear();

	std::string line;
	std::getline(trainingDataFile, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;
	if (label.compare("out:") == 0) {
		double oneValue;
		while (ss >> oneValue) {
			targetOutputVals.push_back(oneValue);
		}
	}

	return targetOutputVals.size();
}
