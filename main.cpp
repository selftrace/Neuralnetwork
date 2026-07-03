#include <iostream>
#include <vector>
#include <fstream>
#include "networkSetUp/Neuron.hpp"
#include "networkSetUp/NetworkLayer.hpp"
#include "networkSetUp/Network.hpp"

int main() {
    std::string fileName;
    std::cout << "Enter filename" << std::endl;
    std::cin >> fileName;

    std::ifstream file(fileName);
    std::vector<std::vector<double>> vectorData;
    std::string dataset;

    if (!file.is_open()) {
        std::cerr << "Couldn't open file.\n";
        return 1;
    }
    
    return 0;
}