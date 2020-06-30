
#include <cstdlib>
#include <string>
#include "perceptron.h"


void printSeparationLine()
{
	printf("############################################################\n");
}

void printFVec(const std::vector<float> &data)
{
	printf(" [");
	if(data.size())
	{
		printf("%f", data[0]);
		for(auto it = data.begin()+1; it != data.end(); ++it)
		{
			printf(",%f", *it);
		}
	}
	printf("]");
}

template<typename T>
void printFVec(const std::vector<std::vector<T>> &data)
{
	printf("\t[");
	if(data.size())
	{
		printFVec(data[0]);
		for(auto it = data.begin()+1; it != data.end(); ++it)
		{
			printf(",");
			printFVec(*it);
		}
	}
	printf("]");
}

void print_pnds(const std::vector<rbnns::perceptron::perceptron_neuron_data> &pnds)
{
	printf("{ ");
	if( pnds.size())
	{
		printf("(th = %f,", pnds[0].theta);
		printFVec(pnds[0].weights);
		printf(")");
	}
	for( size_t i = 1; i < pnds.size(); ++i)
	{
		printf(", (th = %f,", pnds[i].theta);
		printFVec(pnds[i].weights);
		printf(")");
	}
	printf(" }");
}

void testNeuron()
{
	using namespace rbnns;
	
	std::vector<float> neuronTestInput = { .1f, .2f, 0.f, 1.f, 0.01f};
	std::vector<float> neuronTestWeights = { 1.f, 4.f, 1.f, 0.15f, 67.f };
	
	printf( "creating a test neuron with weights : ");
	printFVec(neuronTestWeights);
	printf(" }\n");
	neuron testNeuron( neuronTestWeights, 0.f, 0.f);
	printf("neuron created");
	
	printf("testing neuron with the input of : ");
	printFVec(neuronTestInput);
	printf("\n");
	
	printf("neuron::process result is : %f\n" ,testNeuron.process(neuronTestInput.begin(), neuronTestInput.end()));
}



void test1LayerPerseptron()
{
	using namespace rbnns;
	using pnd_t = perceptron::perceptron_neuron_data;
	std::vector<pnd_t> pnds;
	pnds.emplace_back(std::vector<float>{.1f, .5f, 2.f, 0.04f}, .0f);
	pnds.emplace_back(std::vector<float>{.1f, .5f, 2.f, 0.1f}, .0f);
	pnds.emplace_back(std::vector<float>{.4f, .1f, 5.f, 0.07f}, .0f);
	
	
	printf("creating peceptron with this layers : \n\t");
	print_pnds(pnds);
	printf("\n");
	
	perceptron testpct(.0,{std::move(pnds)});
	
	printf("perceptron created\n");
	
	std::vector<float> neuronTestInput = { .1f, .2f, 0.f, 1.f};
	printf("testing perceptron::process with the input of : ");
	printFVec(neuronTestInput);
	printf("\n");
	std::vector<float> out;
	testpct.process(neuronTestInput, out);
	
	printf("output is : ");
	printFVec(out);
	printf("\n");
	
	std::vector<float> expectedOutput = { 0.1, 0.1, 0.5 };
	printSeparationLine();
	printf("testing perceptron::totalError with the same input and expectedOutput of : ");
	printFVec(expectedOutput);
	printf("\n");
	
	std::vector<float> partialErrors;
	float totalError1 = testpct.totalError(neuronTestInput, expectedOutput, partialErrors);
	printf("total error is %f and the partial errors are : ", totalError1);
	printFVec(partialErrors);
	printf("\n");
	
	std::vector<std::vector<std::vector<float>>> errorGradients;
	
	std::vector<std::vector<float>> out2;
	printSeparationLine();
	printf("testing perceptron::totalErrorAndGradients with the same input and expected output\n");
	float totalError2 = testpct.totalErrorAndGradients(neuronTestInput, expectedOutput, partialErrors, errorGradients, out2);
	printf("total error is : %f\n", totalError2);
	printf("output has %llu elemens and it's print out is is : \n \t", out2.size());
	printFVec(out2);
	printf("\n");
	printf("partial errors are : ");
	printFVec(partialErrors);
	printf("\n");
	printf("error gradients are : ");
	printFVec(errorGradients);
	printf("\n");
}



/*
 * 
 */
int main(int argc, char** argv) {
	
	testNeuron();
	printSeparationLine();
	printSeparationLine();
	test1LayerPerseptron();
	
	return 0;
}

