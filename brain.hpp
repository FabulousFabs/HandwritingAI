//
//  brain.hpp
//  CppCNNHandwriting
//
//  Created by Fabian Schneider on 24.04.19.
//  Copyright © 2019 Fabian Schneider. All rights reserved.
//

#ifndef brain_hpp
#define brain_hpp

#include <iostream>
#include <stdarg.h>
#include <math.h>
#include <algorithm>
#include <vector>

namespace brain {
    //std::vector<int> MakeCircuitVector(int n_args, ...);
    float MakeRandomP();
    float MakeRandomN();
    float MakeRandomNP();
    
    enum ACTIVATION_FUNC
    {
        ACTIVATION_SIGMOID = 0,
        ACTIVATION_TANH,
        ACTIVATION_SOFTPLUS,
        ACTIVATION_RELU
    };
    
    float ActivationSigmoid(float in);
    float DerivativeSigmoid(float in);
    
    float ActivationTanh(float in);
    float DerivativeTanh(float in);
    
    float ActivationSoftplus(float in);
    float DerivativeSoftplus(float in);
    
    float ActivationReLu(float in);
    float DerivativeReLu(float in);
    
    float Activate(float in, enum ACTIVATION_FUNC af);
    float Derive(float in, enum ACTIVATION_FUNC af);
    
    std::vector<std::vector<float>> MatrixDot(std::vector<std::vector<float>> m1, std::vector<std::vector<float>> m2);
    void MatrixFill(bool r, float f, int u, std::vector<std::vector<float>> &m);
    void MatrixOnes(std::vector<std::vector<float>> &m1, std::vector<std::vector<float>> &m2);
    void MatrixZeroes(std::vector<std::vector<float>> &m1, std::vector<std::vector<float>> &m2);
    void MatrixFit(std::vector<std::vector<float>> &m1, std::vector<std::vector<float>> &m2);
    std::vector<std::vector<float>> MatrixT(std::vector<std::vector<float>> &m);
}

#include "brain_cnn.hpp"

#endif /* brain_hpp */
