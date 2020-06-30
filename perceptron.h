

#pragma once
#include "neuron.h"
#include "differential.h"
#include <set>

namespace rbnns
{
	struct perceptron
	{
		struct perceptron_neuron_data  
		{
			perceptron_neuron_data( neuron::weightVector_t &&wgs, float th = 0)
				: weights(std::move(wgs)), theta(th)
			{
				
			}
			neuron::weightVector_t weights;
			float theta = 0;
		};
		
		
		perceptron(float phi, std::vector<std::vector<perceptron_neuron_data>> && pcdsArr)
			: _phi(phi)
		{
			if(!pcdsArr.size())
				throw std::runtime_error("cannot create a perceptron with less than one layer");
			size_t layerCount = pcdsArr.size();
			_neuronsArr.resize(pcdsArr.size());
			for( size_t i = 0; i < layerCount; ++i )
			{
				auto & pcds = pcdsArr[i];
				auto & targetLayer = _neuronsArr[i];
				if( _maxLayerSize < pcds.size())
					_maxLayerSize = pcds.size();
				for( auto & pcd : pcds)
				{
					targetLayer.emplace_back(std::move(pcd.weights), pcd.theta, phi);
				}
			}
		}
		
		
		void process( const neuron::inVector_t &input, std::vector<float> &out)
		{
			
			std::vector<float> oldOut = input;
			size_t prevLayerSize = input.size();
			
			// for performance
			oldOut.resize( _maxLayerSize > input.size() ? _maxLayerSize : input.size());
			out.resize(_maxLayerSize);
			
			
			for( auto &layerNeurons : _neuronsArr )
			{
				size_t layerSize = layerNeurons.size();
				out.resize(layerSize);
				for( size_t i = 0; i < layerSize; ++i)
				{
					out[i] = layerNeurons[i].process( oldOut.begin(),  oldOut.begin() + prevLayerSize );
				}
				prevLayerSize = layerSize;
				oldOut.swap( out );
			}
			
			out.swap(oldOut);
		}
		
		void process( const neuron::inVector_t &input, std::vector<std::vector<float>> &out)
		{
			
			std::vector<float> oldOut = input;
			size_t layerCount = _neuronsArr.size();
			size_t firstLayerSize = _neuronsArr[0].size();
			out.resize(layerCount);
			out[0].resize(firstLayerSize);
			for(size_t i = 0; i < firstLayerSize; ++i)
			{
				out[0][i] = _neuronsArr[0][i].process( input.begin(), input.end());
			}
			for(size_t i = 1; i < layerCount; ++i)
			{
				size_t layerSize = _neuronsArr[i].size();
				out[i].resize(layerSize);
				for(size_t j = 0; j < layerSize; ++j)
				{
					out[i][j] = _neuronsArr[i][j].process( out[i - 1].begin(), out[i - 1].end());
				}
			}
		}
		
		float totalError( const neuron::inVector_t &input, const std::vector<float> &expectedOutput, std::vector<float> &partialError )
		{
			size_t outputCount = _neuronsArr.back().size();
			if(expectedOutput.size() != outputCount)
				throw std::runtime_error("invalid expected output element count");
			process(input, partialError);
			float err = 0;
			for(size_t i = 0; i < outputCount; ++i)
			{
				float tmp = (partialError[i] - expectedOutput[i]);
				tmp = tmp*tmp;
				partialError[i] = tmp;
				err += tmp;
			}
			return err;
		}
		
		float totalErrorAndGradients( const neuron::inVector_t &input, const std::vector<float> &expectedOutput, 
		std::vector<float> &partialError, std::vector<std::vector<std::vector<float>>> &errorGradients, std::vector<std::vector<float>> &out)
		{
			// used to calculate and return the total error value
			float totalError = 0;
			
			// access the gradient of Kth weight (#0th is for "-theta") of the Jth neuron of the Ith layer with errorGradients[I][J][K]
			size_t layerCount = _neuronsArr.size();
			errorGradients.resize(layerCount);
			process(input, out); // calculate the outputs
			std::vector<float> deltas; // delta values for a layer
			std::vector<float> succDeltas;
			deltas.resize(_maxLayerSize);
			succDeltas.resize(_maxLayerSize);
			size_t outputCount = _neuronsArr.back().size();
			for(size_t i = 0; i < outputCount; ++i)
			{
				partialError[i] = (expectedOutput[i] - out.back()[i]);
			}
			
			// calulcating deltas for the last Layer
			auto &outputLayerNeurons = _neuronsArr.back(); // neurons
			auto &outputLayerGradients = errorGradients.back(); // gradients
			outputLayerGradients.resize(outputCount); // outputs
			auto &outputLayerOutputs = out.back();
			for(size_t i = 0; i < outputCount; ++i)
			{
				auto &targetGradients = outputLayerGradients[i];
				size_t wgCount = outputLayerNeurons[i]._weights.size();
				targetGradients.resize(1+wgCount, 0);
				auto &inputLayer = layerCount == 1 ? input : out[layerCount - 2];
				
				// calculating delta
				float delta = partialError[i]*2*outputLayerOutputs[i]*(1-outputLayerOutputs[i]);
				deltas[i] = delta;
				
				// calculating gradient.
				targetGradients[0] = -2*delta;
				size_t inputCount = inputLayer.size();
				
				for(size_t j = 0; j < inputCount; ++j)
				{
					targetGradients[1+j] = -2*delta*inputLayer[j];
				}
			}
			
			for(int64_t layerIndex = layerCount - 2; layerIndex >= 0; --layerIndex)
			{
				deltas.swap(succDeltas);
				auto &layerNeurons = _neuronsArr[layerIndex]; // layer
				auto &nextLayerNeurons = _neuronsArr[layerIndex + 1];
				auto &layerGradients = errorGradients[layerIndex]; // gradients
				auto &layerOutputs = out[layerIndex]; // outputs
				auto &inputLayer = layerIndex == 0 ? input : out[layerIndex - 1];
				
				size_t neuronCount = layerNeurons.size();
				size_t succLayerNCount = nextLayerNeurons.size();
				layerGradients.resize(neuronCount);
				
				for(size_t neuronIndex = 0; neuronIndex < neuronCount; ++neuronIndex)
				{
					size_t wgCount = layerNeurons[neuronIndex]._weights.size();
					auto &targetGradients = layerGradients[neuronIndex];
					targetGradients.resize(wgCount+1, 0);
					
					// calculating delta
					float delta = 0;
					for( size_t succNeuronIndex = 0; succNeuronIndex < succLayerNCount; ++succNeuronIndex)
					{
						delta += succDeltas[succNeuronIndex]*nextLayerNeurons[succNeuronIndex]._weights[neuronIndex];
					}
					float out = layerOutputs[neuronIndex];
					delta *= out*(1-out);
					deltas[neuronIndex] = delta;
					
					// calculating gradients
					targetGradients[0] = -2*delta;
					size_t inputCount = inputLayer.size();
				
					for(size_t j = 0; j < inputCount; ++j)
					{
						targetGradients[1+j] = -2*delta*inputLayer[j];
					}
				}	
			}
			
			
			for( auto &er : partialError)
			{
				er = er*er;
				totalError += er;
			}
			return totalError;
		}
		
		
	private:


		const float _phi = 0;
		std::vector<std::vector<neuron>> _neuronsArr;
		size_t _maxLayerSize = 0;
	};
}
