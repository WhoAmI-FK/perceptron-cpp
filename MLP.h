#pragma once
#include <algorithm>
#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <cmath>
#include <time.h>

namespace neural_network {
	namespace {
		double frand() {
			return (2.0 * static_cast<double>(rand()) / RAND_MAX) - 1.0;
		}
	}
	class Peceptron {
	public:
		std::vector<double> _weights;
		double _bias;
		Peceptron(int inputs, double bias = 1.0)
			: _bias(bias)
		{
			_weights.resize(inputs + 1);
			std::generate(_weights.begin(), _weights.end(),frand);
		}
		double sigmoid(double x);
		void set_weights(std::vector<double> w_init);
		double run(std::vector<double> x) {
			x.push_back(_bias);
			double sum = std::inner_product(x.begin(), x.end(), _weights.begin(), (double)0.0);
			return sigmoid(sum);
		}

	};

	class MultiLayerPeceptron {
	public:
		MultiLayerPeceptron(std::vector<int> layers, double bias = 1.0, double eta = 0.5);
		void set_weights(std::vector<std::vector<std::vector<double>>> w_init);
		std::vector<int> _layers;
		double _bias;
		double _eta;
		std::vector<std::vector<Peceptron>> _network;
		std::vector<std::vector<double>> _values;
		std::vector<std::vector<double>> d;
	};
}