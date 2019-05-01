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
#include <random>

namespace brain {
    //std::vector<int> MakeCircuitVector(int n_args, ...);
    float MakeRandomP();
    float MakeRandomN();
    float MakeRandomNP();
    float MakeRandomXavier(int ins, int outs);
    
    enum WEIGHTS_INIT
    {
        WEIGHTS_INIT_RANDOM = 0,
        WEIGHTS_INIT_XAVIER
    };
    
    enum ACTIVATION_FUNC
    {
        ACTIVATION_SIGMOID = 0,
        ACTIVATION_TANH,
        ACTIVATION_SOFTPLUS,
        ACTIVATION_RELU,
        ACTIVATION_ELU,
        ACTIVATION_RELU_LEAKY
    };
    
    std::vector<std::vector<float>> ActivationSigmoid(std::vector<std::vector<float>> ins);
    float ActivationSigmoid(float in);
    std::vector<std::vector<float>> DerivativeSigmoid(std::vector<std::vector<float>> ins);
    float DerivativeSigmoid(float in);
    
    float ActivationTanh(float in);
    std::vector<std::vector<float>> ActivationTanh(std::vector<std::vector<float>> ins);
    float DerivativeTanh(float in);
    std::vector<std::vector<float>> DerivativeTanh(std::vector<std::vector<float>> ins);
    
    float ActivationSoftplus(float in);
    std::vector<std::vector<float>> ActivationSoftplus(std::vector<std::vector<float>> ins);
    float DerivativeSoftplus(float in);
    std::vector<std::vector<float>> DerivativeSoftplus(std::vector<std::vector<float>> ins);
    
    float ActivationReLu(float in);
    std::vector<std::vector<float>> ActivationReLu(std::vector<std::vector<float>> ins);
    float DerivativeReLu(float in);
    std::vector<std::vector<float>> DerivativeReLu(std::vector<std::vector<float>> ins);
    
    float ActivationELU(float in);
    std::vector<std::vector<float>> ActivationELU(std::vector<std::vector<float>> ins);
    float DerivativeELU(float in);
    std::vector<std::vector<float>> DerivativeELU(std::vector<std::vector<float>> ins);
    
    float ActivationReLuLeaky(float in);
    std::vector<std::vector<float>> ActivationReLuLeaky(std::vector<std::vector<float>> ins);
    float DerivativeReLuLeaky(float in);
    std::vector<std::vector<float>> DerivativeReLuLeaky(std::vector<std::vector<float>> ins);
    
    float Activate(float in, enum ACTIVATION_FUNC af);
    std::vector<std::vector<float>> Activate(std::vector<std::vector<float>> ins, enum ACTIVATION_FUNC af);
    float Derive(float in, enum ACTIVATION_FUNC af);
    std::vector<std::vector<float>> Derive(std::vector<std::vector<float>> ins, enum ACTIVATION_FUNC af);
    
    std::vector<std::vector<float>> MatrixDot(std::vector<std::vector<float>> &m1, std::vector<std::vector<float>> &m2);
    void MatrixFill(bool r, float f, int u, std::vector<std::vector<float>> &m);
    void MatrixOnes(std::vector<std::vector<float>> &m1, std::vector<std::vector<float>> &m2);
    void MatrixZeroes(std::vector<std::vector<float>> &m1, std::vector<std::vector<float>> &m2);
    void MatrixFit(std::vector<std::vector<float>> &m1, std::vector<std::vector<float>> &m2);
    std::vector<std::vector<float>> MatrixT(std::vector<std::vector<float>> &m);
}

#include "brain_cnn.hpp"

#endif /* brain_hpp */
