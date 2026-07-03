#pragma once

#include <vector>

struct Neuron {
    private:
        std::vector<double> weights;
        double bias;
        double delta;
        double output;

        void setOutput(double output);

    public:
        Neuron(int inputNumber);
        std::vector<double> getWeights() const;
        double getDelta() const;
        double getOutput() const;
        void setDelta(double error);
        double calculateWeightedSum(const std::vector<double>& inputs) const;
        double activate(const std::vector<double>& inputs);
        void updateWeights(int idx, double adjsment, bool add);
        void updateBiases(double learnRate, double delta);
};