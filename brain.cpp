//
//  brain.cpp
//  CppCNNHandwriting
//
//  Created by Fabian Schneider on 24.04.19.
//  Copyright © 2019 Fabian Schneider. All rights reserved.
//

#include "brain.hpp"

using namespace brain;

std::vector<int> brain::MakeCircuitVector(int n_args, ...) {
    std::vector<int> v;

    va_list ap;
    va_start(ap, n_args);
        
    for (int i = 1; i <= n_args; i++) {
        int a = va_arg(ap, int);
        v.push_back(a);
    }
    
    va_end(ap);
    
    return v;
}

float brain::MakeRandomP() {
    float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    return r;
}

float brain::MakeRandomN() {
    float r = MakeRandomP() * (-1);
    return r;
}

float brain::MakeRandomNP() {
    float r1 = MakeRandomP(), r2 = MakeRandomP();
    
    if (r2 < 0.5) {
        r1 = r1 * (-1);
    }
    
    return r1;
}

float brain::ActivationSigmoid(float in) {
    float out = 1 / (1 + exp(-in));
    return out;
}

float brain::DerivativeSigmoid(float in) {
    float out = ActivationSigmoid(in) * (1 - ActivationSigmoid(in));
    return out;
}

float brain::ActivationTanh(float in) {
    float out = (2 / (1 + exp(-2 * in))) - 1;
    return out;
}

float brain::DerivativeTanh(float in) {
    float out = 1 - (ActivationTanh(in) * ActivationTanh(in));
    return out;
}

float brain::ActivationSoftplus(float in) {
    float out = log(1 + exp(in));
    return out;
}

float brain::DerivativeSoftplus(float in) {
    float out = 1 / (1 + exp(-in));
    return out;
}

float brain::ActivationReLu(float in) {
    float out = std::max((float) 0, in);
    return out;
}

float brain::DerivativeReLu(float in) {
    float out = (in <= 0) ? (float) 0 : (float) 1;
    return out;
}
