#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

class FileReader {
    private: 
        std::string fileName;
        std::vector<std::vector<double>> dataset;
        std::string data;
    
    public: 
        FileReader(std::string name) {
            fileName = name;
            std::ifstream file(fileName);
        }

        void Read() {
            std::ifstream file(fileName);
            if (!file.is_open()) {
                std::cerr << fileName << " cannot be opened\n";
                return;
            }

            while (std::getline(file, data)) {
                std::stringstream ss(data);
                std::vector<double> dataRow;
                double val;
                char comma;

                while (ss >> val) {
                    dataRow.push_back(val);
                    ss >> comma;
                }
                if (!data.empty()) {
                    dataset.push_back(dataRow);
                }
            }
        }

        void printDataset() {
            for (const auto& dataRow : dataset) {
                for (double val : dataRow) {
                    std::cout << val << " ";
                }
                std::cout << "\n";
            }
        }
};