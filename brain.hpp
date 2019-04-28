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

namespace brain {
    std::vector<int> MakeCircuitVector(int n_args, ...);
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
}

#include "brain_cnn.hpp"

#endif /* brain_hpp */
