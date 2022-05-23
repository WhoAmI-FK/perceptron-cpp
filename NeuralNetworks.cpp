#include <iostream>
#include "MLP.h"

int main()
{
	srand(time(NULL));
	rand();

	std::cout << "\n\n\tLogic Gate\t\n\n";
	neural_network::Peceptron* p = new neural_network::Peceptron(2);
	p->set_weights({ 10,10,-15 });
	std::cout << "AND Gate: " << std::endl;
	std::cout << p->run({ 0,0 }) << std::endl;
	std::cout << p->run({ 0,1 }) << std::endl;
	std::cout << p->run({ 1,0 }) << std::endl;
	std::cout << p->run({ 1,1 }) << std::endl;
	neural_network::Peceptron* p2 = new neural_network::Peceptron(2);
	p2->set_weights({15,15,-10});
	std::cout << "OR Gate: " << std::endl;
	std::cout << p2->run({ 0,0 }) << std::endl;
	std::cout << p2->run({ 0,1 }) << std::endl;
	std::cout << p2->run({ 1,0 }) << std::endl;
	std::cout << p2->run({ 1,1 }) << std::endl;
	neural_network::Peceptron* p3 = new neural_network::Peceptron(2);
	p3->set_weights({ -15,-15,10 });
	std::cout << "NOR Gate: " << std::endl;
	std::cout << p3->run({ 0,0 }) << std::endl;
	std::cout << p3->run({ 0,1 }) << std::endl;
	std::cout << p3->run({ 1,0 }) << std::endl;
	std::cout << p3->run({ 1,1 }) << std::endl;
	neural_network::Peceptron* p4 = new neural_network::Peceptron(2);
	p4->set_weights({ -10,-10,15 });
	std::cout << "NAND Gate: " << std::endl;
	std::cout << p4->run({ 0,0 }) << std::endl;
	std::cout << p4->run({ 0,1 }) << std::endl;
	std::cout << p4->run({ 1,0 }) << std::endl;
	std::cout << p4->run({ 1,1 }) << std::endl;

	std::cout << "\n\nHardCoded XOR: " << std::endl;
	neural_network::MultiLayerPeceptron mlp = neural_network::MultiLayerPeceptron({ 2,2,1 });
	mlp.set_weights({ {{-10,-10,15},{15,15,-10}},{{10,10,-15}} });
	std::cout << "Weights:\n";
	mlp.print_weights();

	std::cout << "XOR: " << std::endl;
	std::cout << "0 0 = " << mlp.run({ 0,0 })[0] << std::endl;
	std::cout << "0 1 = " << mlp.run({ 0,1 })[0] << std::endl;
	std::cout << "1 0 = " << mlp.run({ 1,0 })[0] << std::endl;
	std::cout << "1 1 = " << mlp.run({ 1,1 })[0] << std::endl;


	//test code - Trained XOR
	std:: cout << "\n\n--------Trained XOR Example----------------\n\n";
	mlp = neural_network::MultiLayerPeceptron({ 2,2,1 });
	std::cout << "Training Neural Network as an XOR Gate...\n";
	double MSE;
	for (std::size_t i = 0; i < 3000; i++) {
		MSE = 0.0;
		MSE += mlp.bp({ 0,0 }, { 0 });
		MSE += mlp.bp({ 0,1 }, { 1 });
		MSE += mlp.bp({ 1,0 }, { 1 });
		MSE += mlp.bp({ 1,1 }, { 0 });
		MSE = MSE / 4.0;
		if (i % 100 == 0)
			std::cout << "MSE = " << MSE << std::endl;
	}

	std::cout << "\n\nTrained weights (Compare to hard-coded weights):\n\n";
	mlp.print_weights();

	std::cout << "XOR:" << std::endl;
	std::cout << "0 0 = " << mlp.run({ 0,0 })[0] << std::endl;
	std::cout << "0 1 = " << mlp.run({ 0,1 })[0] << std::endl;
	std::cout << "1 0 = " << mlp.run({ 1,0 })[0] << std::endl;
	std::cout << "1 1 = " << mlp.run({ 1,1 })[0] << std::endl;


	delete p, p2,p3,p4;
	return 0;
} 