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
	delete p, p2,p3,p4;
	return 0;
}