#include "mathFunc.h"
#include "Neuron.hpp"
#include <vector>
#include <optional>

/*This struct is supposed to represent artificial neurons in a network,
-take multiple inputs
-processing them by multiplying with weight and then adding bias to weighted sum
*/

void Neuron::setOutput(double output) {
    this->output = output;
};

Neuron::Neuron(int inputNumber) : delta(0.0), output(0.0) {
    this->weights.resize(inputNumber);
}

std::vector<double> Neuron::getWeights() const { return this->weights; };
double Neuron::getDelta() const { return delta; };
double Neuron::getOutput() const { return output; };

void Neuron::setDelta(double error) {
    /*the product of the error signal from the next layer multipled with our neuron's output gives us
    the gradient to update weights during training, and the weight itself comes in layer
    when computing the gradient for updating 
    -> ∂L/∂y(error term) * ∂y/∂z(activation function) * ∂z/∂wi(weight)*/
    delta = error * reluDerivative(output);
}

double Neuron::calculateWeightedSum(const std::vector<double>& inputs) const {
    /*Each input multipled with corresponding weight, then all products are added together and added with the bias to return total sum*/
    double sum = this->bias;
    for (int i = 0; i < inputs.size(); i++) {
        sum += weights[i] * inputs[i];
    }
    return sum;
}

double Neuron::activate(const std::vector<double>& inputs) {
    /*The weighted sum is taken and thrown into relu to either have a positive value or 0
    forward propagation is prediction, backpropagation is learning to output the final value*/
    this->output = relu(calculateWeightedSum(inputs));
    setOutput(output);
    return output;
}

void Neuron::updateWeights(int idx, double adjsment, bool add) {
    if (add) {
        weights[idx] += adjsment;
        return;
    }
    weights[idx] = adjsment;
}

void Neuron::updateBiases(double learnRate, double delta) {
    bias += learnRate * delta;
}