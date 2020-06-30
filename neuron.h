#pragma once

#include <functional>
#include <vector>
#include <cmath>
#include <exception>
namespace rbnns
{
	struct neuron
	{
		using inVector_t = std::vector<float /*input*/>;
		using weightVector_t = inVector_t;
		
		neuron( weightVector_t &&weights, float theta, float phi)
			: _weights(std::move(weights)), _theta(theta), _phi(phi)
		{
		}
		
		neuron( weightVector_t &weights, float theta, float phi)
			: _weights(weights), _theta(theta), _phi(phi)
		{
		}
		
		static constexpr float fout(float act) { return 1; }
		template<typename itT>
		static float fnet(itT inputBegin, itT inputEnd, const weightVector_t &weights, float theta) 
		{ 
			float sum = -theta;
			size_t totalSize = inputEnd - inputBegin;
			if(totalSize > weights.size())
			{
				throw std::runtime_error("input count greater than weight count");
			}

			for(auto wgit = weights.begin(); inputBegin != inputEnd; ++inputBegin)
			{
				sum += (*inputBegin) * (*wgit);
			}

			return sum;
		}
		static constexpr float fact(float net, float phi)
		{
			return 2/(1 + std::exp(phi - net)) - 1;
		}
		
		
		template<typename itT>
		float process(itT inputBegin, itT inputEnd)
		{
			float out = fnet(inputBegin, inputEnd, _weights, _theta);
			out = fact(out, _phi);
			//out = fout(out); not needed
			return out;
		}
		
		float _phi = 0.0f;
		float _theta = 1.0f;
		weightVector_t _weights;
	};
	
}