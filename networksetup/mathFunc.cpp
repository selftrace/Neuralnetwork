#include "mathFunc.h"
#include <algorithm>

/*The reason why I chose ReLu is because it ensures that the network is activated
if the input value is positive, otherwise its 0 if negative to prevent computational
inefficiency- making it faster than sigmoid or tanh*/
double relu(double x) {
    /*This ensures so that if x is negative, to just return 0, otherwise, return x*/
    return std::max(0.0, x);
}

double reluDerivative(double x) {
    /*This is for backpropagation so that the neuron essentially turns off if x is <= 0*/
    return (x > 0) ? 1.0 : 0.0;
}