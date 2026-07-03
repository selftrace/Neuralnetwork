#pragma once

#include <vector>
#include "Neuron.hpp"
#include "NetworkLayer.hpp"

class NetworkLayer {
    private:
        std::vector<Neuron> neuronLayer;
    
    public:
        NetworkLayer(std::vector<Neuron>&& neurons);

        std::vector<Neuron>& getLayerNeurons();
        std::vector<double> getTotalOutputs() const;
        std::vector<double> activateNeurons(const std::vector<double>& inputs);
        double calculateTotalDelta(int previousNeuronIndex) const;

        
};