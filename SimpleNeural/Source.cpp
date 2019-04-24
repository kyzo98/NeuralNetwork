#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include "Net.h"

void ShowVectorVals(std::string label, std::vector<double> &v)
{
	std::cout << label << " ";
	for (unsigned i = 0; i < v.size(); i++)
	{
		std::cout << v[i] << " ";
	}

	std::cout << std::endl;
}

void ShowVectorValsRounded(std::string label, std::vector<double> &v)
{
	std::cout << label << " ";
	for (unsigned i = 0; i < v.size(); i++)
	{
		std::cout << std::fixed << std::setprecision(1) << v[i] << " ";
	}

	std::cout << std::endl;
}

int main() 
{
	TrainingData trainData("trainingData.txt");

	// e. g., { 3, 2, 1 }
	std::vector<unsigned> topology;
	trainData.GetTopology(topology);
	Net myNet(topology);
	
	std::vector<double> inputVals, targetVals, resultVals;
	int trainingPass = 0;
	int netFunc = 0;
	std::cout << std::endl << "Train [0] / Test [1] / TestRandom[2]" << std::endl;
	std::cin >> netFunc;

	bool anotherTest;

	switch (netFunc) {
	case 0:	//Train net
		while (!trainData.IsEof())
		{
			//Get new input data and feed it forward
			if (trainData.GetNextInputs(inputVals) != topology[0]) {
				break;
			}

			++trainingPass;
			std::cout << std::endl << "Pass " << trainingPass;


			ShowVectorVals(": Inputs:", inputVals);
			myNet.FeedForward(inputVals);

			//Collect the net's actual results
			myNet.GetResults(resultVals);
			ShowVectorVals(": Outputs:", resultVals);

			//Train the net what the outputs should have been
			trainData.GetTargetOutputs(targetVals);
			ShowVectorVals("Targets:", targetVals);
			assert(targetVals.size() == topology.back());

			myNet.BackProp(targetVals);

			//Report how well the training is working averaged over recent samples
			std::cout << "net recent average error: " << myNet.GetRecentAverageError() << std::endl;
		}

		char c;
		std::cout << std::endl << "Save Net? [y/n]" << std::endl;
		std::cin >> c;
		if (c == 'y') {
			myNet.SaveNet();
		}
		break;
	case 1: //Test net
		anotherTest = true;

		//Load data
		myNet.LoadNet();

		while (anotherTest) {
			//Set new inputs
			inputVals.clear();

			for (unsigned i = 0; i < topology[0]; i++)
			{
				int input;
				std::cin >> input;

				inputVals.push_back(input);
			}

			//Test net with nex inputs
			ShowVectorVals(": Inputs:", inputVals);
			myNet.FeedForward(inputVals);

			//Collect the net's actual results
			myNet.GetResults(resultVals);
			ShowVectorValsRounded(": Outputs:", resultVals);

			char c;
			std::cout << std::endl << "Another test? [y/n]" << std::endl;
			std::cin >> c;
			if (c == 'n') {
				anotherTest = false;
			}
		}
		break;
	case 2:
		anotherTest = true;

		while (anotherTest) {
			//Set new inputs
			inputVals.clear();

			for (unsigned i = 0; i < topology[0]; i++)
			{
				int input;
				std::cin >> input;

				inputVals.push_back(input);
			}

			//Test net with nex inputs
			ShowVectorVals(": Inputs:", inputVals);
			myNet.FeedForward(inputVals);

			//Collect the net's actual results
			myNet.GetResults(resultVals);
			ShowVectorValsRounded(": Outputs:", resultVals);

			char c;
			std::cout << std::endl << "Another test? [y/n]" << std::endl;
			std::cin >> c;
			if (c == 'n') {
				anotherTest = false;
			}
		}
		break;
	default:
		std::cout << "netFunc not aviable.";
		break;
	}

	system("pause");
	return 0;
}