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
}

void brain::layer::flatten_proto::Excite(brain::layer_proto *next_layer) {
    std::vector<std::vector<float>> neuronsOut;
    neuronsOut.push_back(this->v_fNeurons);
    this->v_fNeuronsOut = neuronsOut[0];
    
    for (int i = 0; i < this->v_fWeights[0].size() - 1; i++) {
        neuronsOut.push_back(this->v_fNeuronsOut);
    }
    neuronsOut = neuronsOut; // i x j
    std::vector<std::vector<float>> weights = this->v_fWeights; // j x i
    
    std::vector<std::vector<float>> p = brain::MatrixDot(neuronsOut, weights); // i x i (i times the output)
    
    next_layer->v_fNeurons = p[0];
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

void brain::layer_proto::Neuroplasticity(enum brain::WEIGHTS_INIT wi, int ins, int outs) {
    for (int i = 0; i < this->v_fNeurons.size(); i++) {
        std::vector<float> p;
        for (int j = 0; j < outs; j++) {
            float p1 = (wi == WEIGHTS_INIT_RANDOM) ? brain::MakeRandomNP() : brain::MakeRandomXavier(ins, outs);
            p.push_back(p1);
        }
        this->v_fWeights.push_back(p);
    }
}

void brain::layer_proto::GetSensations(std::vector<float> s) {
    assert(s.size() == this->v_fNeurons.size());
    
    for (int i = 0; i < s.size(); i++) {
        this->v_fNeurons[i] = s[i];
    }
}

void brain::layer_proto::Excite(brain::layer_proto *next_layer) {
    std::vector<std::vector<float>> neurons;
    neurons.push_back(this->v_fNeurons);
    
    std::vector<std::vector<float>> neuronsOut = brain::Activate(neurons, this->e_iActivationFunc);
    this->v_fNeuronsOut = neuronsOut[0];
    
    for (int i = 0; i < this->v_fWeights[0].size() - 1; i++) {
        neuronsOut.push_back(this->v_fNeuronsOut);
    }
    neuronsOut = neuronsOut; // i x j
    std::vector<std::vector<float>> weights = this->v_fWeights; // j x i
    
    std::vector<std::vector<float>> p = brain::MatrixDot(neuronsOut, weights); // i x i (i times the output)
    
    next_layer->v_fNeurons = p[0];
}

void brain::layer_proto::Activate() {
    for (int i = 0; i < this->v_fNeurons.size(); i++) {
        this->v_fNeuronsOut[i] = brain::Activate(this->v_fNeurons[i], this->e_iActivationFunc);
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

void brain::CNN::Compile(enum brain::WEIGHTS_INIT wi) {
    assert(this->b_IsCompiled == false);
    
    // grow neurons
    for (int i = 0; i < this->Layers.size(); i++) {
        this->Layers[i].Grow();
    }
    
    // neuroplasticity
    for (int i = 0; i < this->Layers.size() - 1; i++) {
        int ins = (i > 0) ? (int) this->Layers[i - 1].v_fNeurons.size() : 0, outs = (int) this->Layers[i + 1].v_fNeurons.size();
        this->Layers[i].Neuroplasticity(wi, ins, outs);
    }
    
    this->b_IsCompiled = true;
}

std::tuple<int, float> brain::CNN::Perceive(std::vector<float> &s) {
    assert(this->b_IsCompiled == true);
    
    this->Layers[0].GetSensations(s);
    this->Layers[0].Excite(&this->Layers[1]);
    
    if (this->Layers.size() > 2) {
        for (int i = 1; i < this->Layers.size() - 1; i++) {
            this->Layers[i].Excite(&this->Layers[i + 1]);
        }
    }
    
    this->Layers[this->Layers.size() - 1].Activate();
    
    return this->GetChoice();
}

std::tuple<int, float> brain::CNN::GetChoice() {
    assert(this->b_IsCompiled == true);
    
    float max = -2;
    int maxN = 0;
    
    for (int i = 0; i < this->Layers[this->Layers.size() - 1].v_fNeuronsOut.size(); i++) {
        if (this->Layers[this->Layers.size() - 1].v_fNeuronsOut[i] > max) {
            max = this->Layers[this->Layers.size() - 1].v_fNeuronsOut[i];
            maxN = i;
        }
    }
    
    return std::tuple<int, float>(maxN, max);
}
