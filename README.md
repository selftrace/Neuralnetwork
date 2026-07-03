# Neuralnetwork

A lightweight, object-oriented feedforward neural network implemented completely from scratch in C++. This project demonstrates the core mathematical mechanics of deep learning, featuring manual implementations of forward propagation, backpropagation and gradient-based optimization without relying on external machine learning libraries.

## The core components
Neuron: Manages its own unique state, containing weights, biases, and error deltas. It calculates its weighted sum and handles activation routines
NetworkLayer: Encapsulates groups of neurons, extracting combined activations and passing aggregated outputs efficiently down the pipeline
Network: The brain of the application. Orchestrates the full lifecycle of training (feedForward, backPropagate, trainModel) using a customizable learning rate.

## Theoretical mechanics implemented
Forward propagation: Computes predictions by scaling inputs by weights, shifting via biases, and filtering through ReLU
Backpropagation: Translates error signals from the output layer back through the hidden layers by multiplying the error term by the derivative of the activation function
Cost function: Monitors network performance via MSE optimization

## Why ReLU?
This network explicitly opts for ReLU over traditional sigmoid or tanh functions
double relu(double x) {
    return std::max(0.0, x);
}
By forcing negative values directly to 0.0, the network ensures sparse activation. This speeds up math operations significantly, limits computational bloat, and provides deterministic gradients (1.0 or 0.0) that mitigate the vanishing gradient problem during backpropagation.
