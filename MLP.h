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
		double sigmoid(double x) {
			return ((std::exp(x)) / (std::exp(x) + 1.0));
		}
		void set_weights(std::vector<double> w_init) {
			_weights.assign(w_init.begin(), w_init.end());
		}
		double run(std::vector<double> x) {
			x.push_back(_bias);
			double sum = std::inner_product(x.begin(), x.end(), _weights.begin(), (double)0.0);
			return sigmoid(sum);
		}

	};

	class MultiLayerPeceptron {
	public:
		MultiLayerPeceptron(std::vector<int> layers, double bias = 1.0, double eta = 0.5) {
			_layers = layers;
			_bias = bias;
			_eta = eta;
			for (std::size_t i = 0; i < layers.size(); i++) {
				_values.push_back(std::vector<double>(layers[i], 0.0));
				d.push_back(std::vector < double>(layers[i], 0.0));
				_network.push_back(std::vector<Peceptron>());
				if (i > 0) {
					for (std::size_t j = 0; j < layers[i]; j++)
					{
						_network[i].push_back(Peceptron(layers[i-1],bias));
					}
				}
			}
		}
		void set_weights(std::vector<std::vector<std::vector<double>>> w_init) {
			for (std::size_t i = 0; i < w_init.size(); i++) {
				for (std::size_t j = 0; j < w_init[i].size(); j++) {
					_network[i+1][j].set_weights(w_init[i][j]);
				}
			}
		}
		void print_weights()
		{
			std::cout << std::endl;
			for (std::size_t i = 1; i < _network.size(); i++)
			{
				for (std::size_t j = 0; j < _layers[i]; j++) {
					std::cout << "Layer " << i + 1 << " Neuron " << j << ": ";
					for (auto& it : _network[i][j]._weights)
					{
						std::cout << it << " ";
					}
					std::cout << std::endl;
				}
			}
			std::cout << std::endl;
		}
		std::vector<double> run(std::vector<double> x) {
			_values[0] = x;
			for (std::size_t i = 1; i < _network.size(); i++) {
				for (std::size_t j = 0; j < _layers[i]; j++) {
					_values[i][j] = _network[i][j].run(_values[i - 1]);
				}
			}
			return _values.back();
		}
		double bp(std::vector<double> x, std::vector<double> y)
		{
			// Backpropagation Step by Step:

			  // STEP 1: Feed a sample to the network
			std::vector<double> outputs = run(x);

			// STEP 2: Calculate the MSE
			std::vector<double> error;
			double MSE = 0.0;
			for (int i = 0; i < y.size(); i++) {
				error.push_back(y[i] - outputs[i]);
				MSE += error[i] * error[i];
			}
			MSE /= _layers.back();

			// STEP 3: Calculate the output error terms
			for (int i = 0; i < outputs.size(); i++)
				d.back()[i] = outputs[i] * (1 - outputs[i]) * (error[i]);

			// STEP 4: Calculate the error term of each unit on each layer    
			for (int i = _network.size() - 2; i > 0; i--)
				for (int h = 0; h < _network[i].size(); h++) {
					double fwd_error = 0.0;
					for (int k = 0; k < _layers[i + 1]; k++)
						fwd_error += _network[i + 1][k]._weights[h] * d[i + 1][k];
					d[i][h] = _values[i][h] * (1 - _values[i][h]) * fwd_error;
				}

			// STEPS 5 & 6: Calculate the deltas and update the weights
			for (int i = 1; i < _network.size(); i++)
				for (int j = 0; j < _layers[i]; j++)
					for (int k = 0; k < _layers[i - 1] + 1; k++) {
						double delta;
						if (k == _layers[i - 1])
							delta = _eta * d[i][j] * _bias;
						else
							delta = _eta * d[i][j] * _values[i - 1][k];
						_network[i][j]._weights[k] += delta;
					}
			return MSE;
		}
		std::vector<int> _layers;
		double _bias;
		double _eta;
		std::vector<std::vector<Peceptron>> _network;
		std::vector<std::vector<double>> _values;
		std::vector<std::vector<double>> d;
	};
}