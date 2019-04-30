//
//  brain_cnn.cpp
//  CppCNNHandwriting
//
//  Created by Fabian Schneider on 30.04.19.
//  Copyright © 2019 Fabian Schneider. All rights reserved.
//

#include "brain_cnn.hpp"

//
//
// CLASS brain::layer::flatten_proto

std::tuple<enum brain::CNN_LAYER_TYPE, int, enum brain::ACTIVATION_FUNC> brain::layer::Flatten(int n_args, ...) {
    va_list ap;
    va_start(ap, n_args);
    
    int neurons = 1;
    
    for (int i = 1; i < n_args; i++) {
        int n = va_arg(ap, int);
        neurons = neurons * n;
    }
    
    return std::tuple<enum brain::CNN_LAYER_TYPE, int, enum brain::ACTIVATION_FUNC>(CNN_LAYER_T_FLATTEN, neurons, ACTIVATION_SIGMOID);
}

brain::layer::flatten_proto::flatten_proto(int n, enum brain::ACTIVATION_FUNC af) {
    this->i_nNeurons = n;
    this->e_iActivationFunc = af;
    
    std::cout << this->i_nNeurons << std::endl;
}

//
//
// CLASS brain::layer::dense_proto

std::tuple<enum brain::CNN_LAYER_TYPE, int, enum brain::ACTIVATION_FUNC> brain::layer::Dense(int neurons, enum brain::ACTIVATION_FUNC af) {
    return std::tuple<enum brain::CNN_LAYER_TYPE, int, enum brain::ACTIVATION_FUNC>(CNN_LAYER_T_FLATTEN, neurons, af);
}

brain::layer::dense_proto::dense_proto(int n, enum brain::ACTIVATION_FUNC af) {
    this->i_nNeurons = n;
    this->e_iActivationFunc = af;
    
    std::cout << this->i_nNeurons << std::endl;
}

//
//
// CLASS brain::layer_proto
void brain::layer_proto::Grow() {
    for (int i = 0; i < this->i_nNeurons; i++) {
        this->v_fNeurons.push_back((float) 0);
        this->v_fNeuronsOut.push_back((float) 0);
    }
}

void brain::layer_proto::Neuroplasticity(int n) {
    for (int i = 0; i < n; i++) {
        float p = brain::MakeRandomNP();
        this->v_fWeights.push_back(p);
    }
}

//
//
// CLASS brain::CNN

brain::CNN::CNN() {
    this->b_IsCompiled = false;
}

void brain::CNN::Sequential(std::tuple<enum brain::CNN_LAYER_TYPE, int, enum brain::ACTIVATION_FUNC> l) {
    assert(this->b_IsCompiled == false);
    
    if (std::get<0>(l) == CNN_LAYER_T_FLATTEN) {
        brain::layer::flatten_proto ls(std::get<1>(l), std::get<2>(l));
        this->Layers.push_back(ls);
    } else {
        brain::layer::dense_proto ls(std::get<1>(l), std::get<2>(l));
        this->Layers.push_back(ls);
    }
}

void brain::CNN::Compile() {
    assert(this->b_IsCompiled == false);
    
    // grow neurons
    for (int i = 0; i < this->Layers.size(); i++) {
        this->Layers[i].Grow();
    }
    
    // neuroplasticity
    for (int i = 0; i < this->Layers.size() - 1; i++) {
        this->Layers[i].Neuroplasticity((int) this->Layers[i + 1].v_fNeurons.size());
    }
    
    this->b_IsCompiled = true;
}

void brain::CNN::Perceive(std::vector<float> &s) {
    
}
