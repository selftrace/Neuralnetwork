#include <vector>
#include "Neuron.hpp"
#include "NetworkLayer.hpp"

NetworkLayer::NetworkLayer(std::vector<Neuron>&& neurons) : neuronLayer(std::move(neurons)) {}

std::vector<Neuron>& NetworkLayer::getLayerNeurons() {
    return neuronLayer;
};

std::vector<double> NetworkLayer::getTotalOutputs() const {
    /*This method extracts all outputs of a layer to input into the next layer for forward propagation, decided to 
    not use auto during iteration and index instead for clear verbosity and more control. Since vectors are typically memory
    inefficient, I made sure to reserve */
    std::vector<double> totalOutput;
    totalOutput.reserve(neuronLayer.size());
    for(size_t i = 0; i < neuronLayer.size(); i++) {
        totalOutput.push_back(neuronLayer[i].getOutput());
    }
    return totalOutput;
};

std::vector<double> NetworkLayer::activateNeurons(const std::vector<double>& inputs) {
    /*All raw neurons are taken and their respective outputs are put in the ReLu function to let it learn non-linear relationships. This
    is key to backpropagation by introducing complex gradients that will be computed. Since ReLu only activates a neuron when the value is
    positive, this ensures the network's efficiency too*/
    std::vector<double> neuronActivations;
    neuronActivations.reserve(neuronLayer.size());
    for(size_t i = 0; i < neuronLayer.size(); i++) {
        neuronActivations.push_back(neuronLayer[i].activate(inputs));
    }
    return neuronActivations;
}

double NetworkLayer::calculateTotalDelta(int previousNeuronIndex) const {
    /*The purpose of this method is to calculate the error to send to the previous layer during backpropagation, it does this by
    getting the value of each edge (weight) between current and previous layer's neurons and multiplying it by the delta. This is
    important because the purpose of backpropagation is to tweak all these nobs and dials of the weights and biases to 
    minimize the value of the cost function since its the only thing in our control as opposed to the direct input and output of each neuron*/
    double totalSum = 0.0;
    for(size_t i = 0; i < neuronLayer.size(); i++) {
        totalSum += neuronLayer[i].getWeights()[previousNeuronIndex] * neuronLayer[i].getDelta();
    }
    return totalSum;
}