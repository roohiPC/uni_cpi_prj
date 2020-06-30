

#pragma once

namespace rbnns
{
	template<typename callableT, typename valueT>
	auto derivative(callableT &&func, valueT x, valueT x_diff = 0.0001)
	{
		return (func(x+x_diff) - func(x))/x_diff;
	}
}
