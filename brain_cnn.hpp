//
//  brain_cnn.hpp
//  CppCNNHandwriting
//
//  Created by Fabian Schneider on 28.04.19.
//  Copyright © 2019 Fabian Schneider. All rights reserved.
//

#ifndef brain_cnn_hpp
#define brain_cnn_hpp

#include <iostream>
#include <vector>

#include "brain.hpp"

namespace brain {
    // convolutional neural network
    class CNN
    {
    private:
        std::vector<std::vector<int>> v_iCircuit;
        std::vector<std::vector<int>> v_iCircuitNet;
        std::vector<std::vector<std::vector<float>>> v_fWeights;
        enum ACTIVATION_FUNC e_iFunction;
        float f_Eta;
        
    public:
        CNN(std::vector<int> &circuit_structure, float learning_rate, enum ACTIVATION_FUNC af);
        int Perceive(std::vector<float> &stimulus);
        void AssumeRestingState();
        void Feedback(int percept, int correct, std::vector<float> &stimulus);
    };
}

#endif /* brain_cnn_hpp */
