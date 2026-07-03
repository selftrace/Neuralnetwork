#include <cmath>
#include <iostream>
#include "Network.hpp"

vector<NetworkLayer> Network::getLayers() const {
    vector<NetworkLayer> allLayers = hiddenLayers;
    allLayers.push_back(outputLayer);
    return allLayers;
}

vector<double> Network::feedForward(vector<double>& inputs) {
    for (auto& i : hiddenLayers){
        inputs = i.activateNeurons(inputs);
    }
    return outputLayer.activateNeurons(inputs);
}

vector<double> Network::errorCalculation(vector<double>& givenValue, vector<double>& expectedValue){
    vector<double> errorDifferences;
    for (size_t i = 0; i < givenValue.size(); i++){
        errorDifferences.push_back(givenValue[i] - expectedValue[i]);
    }
    return errorDifferences;
}

void Network::backPropagate(vector<double>& inputs, vector<double>& errors){
    for (size_t i = 0; i < outputLayer.getLayerNeurons().size(); i++){
        outputLayer.getLayerNeurons()[i].setDelta(errors[i]);
    }

    for (int layerIndex = hiddenLayers.size() - 1; layerIndex >= 0; --layerIndex){
        NetworkLayer& layer = hiddenLayers[layerIndex];
        NetworkLayer& nextLayer = (layerIndex == hiddenLayers.size() - 1) ? outputLayer : hiddenLayers[layerIndex + 1];

        for (size_t i = 0; i < layer.getLayerNeurons().size(); ++i){
            Neuron& neuron = layer.getLayerNeurons()[i];
            double nextError = nextLayer.calculateTotalDelta(i);
            neuron.setDelta(nextError);
        }
    }
    updateAllWeights(inputs);
}

void Network::updateAllWeights(vector<double>& inputs){
    for (size_t layerIndex = 0; layerIndex < hiddenLayers.size(); layerIndex++){
        NetworkLayer& layer = hiddenLayers[layerIndex];
        vector<double> previousOutputs = (layerIndex == 0) ? inputs : hiddenLayers[layerIndex - 1].getTotalOutputs();
        for(Neuron& i : layer.getLayerNeurons()){
            updateSingleWeights(previousOutputs, i);
        }
    }
    vector<double> lastHiddenOutput = hiddenLayers.back().getTotalOutputs();
    for (size_t i = 0; i < outputLayer.getLayerNeurons().size(); i++){
        updateSingleWeights(lastHiddenOutput, outputLayer.getLayerNeurons()[i]);
    }
}

void Network::updateSingleWeights(vector<double>& previousOutputs, Neuron& neuron){
    for (size_t i = 0; i < previousOutputs.size(); i++){
        neuron.updateWeights(i, (rate * neuron.getDelta() * previousOutputs[i]), true);
    }
    neuron.updateBiases(this->rate, neuron.getDelta());  // Moved outside loop
}

double Network::meanSquaredError(vector<double>& prediction, vector<double>& expected){
    double totalError = 0.0;
    for (size_t i = 0; i < prediction.size(); i++){
        totalError += pow((prediction[i] - expected[i]), 2);
    }
    return totalError / prediction.size();
}

void Network::trainModel(int epochNumber, int outputNumber, vector<vector<double>>& trainingSet, vector<double>& trainingOutput){
    for (int epoch = 0; epoch < epochNumber; epoch++){
        double sumError = 0.0;
        
        for (size_t j = 0; j < trainingSet.size(); j++){
            vector<double> expectedValues(outputNumber, 0.0);
            expectedValues[static_cast<int>(trainingOutput[j])] = 1.0;
            
            vector<double> prediction = feedForward(trainingSet[j]);
            vector<double> error = errorCalculation(prediction, expectedValues);
            backPropagate(trainingSet[j], error);
            
            sumError += meanSquaredError(prediction, expectedValues);
        }
        
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Avg Error: " << sumError / trainingSet.size() << std::endl;
        }
    }
}

int Network::predictInfo(vector<double>& inputs) {
    vector<double> allOutputs = feedForward(inputs);
    return std::max_element(allOutputs.begin(), allOutputs.end()) - allOutputs.begin();
}