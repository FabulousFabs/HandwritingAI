//
//  brain_cnn.cpp
//  CppCNNHandwriting
//
//  Created by Fabian Schneider on 28.04.19.
//  Copyright © 2019 Fabian Schneider. All rights reserved.
//

#include "brain_cnn.hpp"

brain::CNN::CNN(std::vector<int> &circuit_structure, float eta = 0.5, enum ACTIVATION_FUNC af = ACTIVATION_TANH) {
    this->f_Eta = eta;
    this->e_iFunction = af;
    
    // setup neurons
    for (int k = 0; k < circuit_structure.size(); k++) {
        std::vector<int> v;
        
        for (int l = 0; l < circuit_structure[k]; l++) {
            v.push_back(0);
        }
        
        // bias neuron
        if (k < circuit_structure.size() - 1) {
            v.push_back(1);
        }
        
        this->v_iCircuit.push_back(v);
    }
    this->v_iCircuitNet = this->v_iCircuit;
    
    // setup weights
    for (int i = 0; i < this->v_iCircuit.size() - 1; i++) {
        std::vector<std::vector<float>> w;
        
        for (int n = 0; n < this->v_iCircuit[i].size(); n++) {
            std::vector<float> w1;
            
            for (int j = 0; j < this->v_iCircuit[i+1].size()-1; j++) {
                float r = brain::MakeRandomNP();
                w1.push_back(r);
            }
            
            w.push_back(w1);
        }
        
        this->v_fWeights.push_back(w);
    }
}

int brain::CNN::Perceive(std::vector<float> &stimulus) {
    this->AssumeRestingState();
    
    // feed sensory neurons
    for (int i = 0; i < stimulus.size(); i++) {
        this->v_iCircuit[0][i] = stimulus[i];
        this->v_iCircuitNet[0][i] = stimulus[i];
    }
    
    // activation
    for (int i = 0; i < this->v_iCircuit.size() - 1; i++) {
        for (int n = 0; n < this->v_iCircuit[i].size(); n++) {
            for (int j = 0; j < this->v_fWeights[i][n].size(); j++) {
                this->v_iCircuit[i + 1][j] += this->v_iCircuit[i][n] * this->v_fWeights[i][n][j];
                this->v_iCircuitNet[i + 1][j] = this->v_iCircuit[i + 1][j];
            }
        }
        
        for (int n = 0; n < this->v_iCircuit[i + 1].size(); n++) {
            this->v_iCircuit[i][n] = (this->e_iFunction == ACTIVATION_SIGMOID) ? brain::ActivationSigmoid(this->v_iCircuit[i][n])
                                    : (this->e_iFunction == ACTIVATION_TANH) ? brain::ActivationTanh(this->v_iCircuit[i][n])
                                    : (this->e_iFunction == ACTIVATION_SOFTPLUS) ? brain::ActivationSoftplus(this->v_iCircuit[i][n])
                                    : brain::ActivationReLu(this->v_iCircuit[i][n]);
        }
    }
    
    // mechanical action
    int perceptN = 0;
    float perceptMax = -1.01;
    
    for (int i = 0; i < this->v_iCircuit[this->v_iCircuit.size() - 1].size(); i++) {
        if (this->v_iCircuit[this->v_iCircuit.size() - 1][i] > perceptMax) {
            perceptN = i;
            perceptMax = this->v_iCircuit[this->v_iCircuit.size() - 1][i];
        }
    }
    
    return perceptN;
}

void brain::CNN::AssumeRestingState() {
    for (int i = 0; i < this->v_iCircuit.size(); i++) {
        for (int n = 0; n < this->v_iCircuit[i].size() - 1; n++) {
            this->v_iCircuit[i][n] = 0;
            this->v_iCircuitNet[i][n] = 0;
        }
    }
}

void brain::CNN::Feedback(int percept, int correct, std::vector<float> &stimulus) {
    if (percept == correct) {
        return;
    }
    
    // calculate errors
    float error_total = 0;
    std::vector<float> error_out;
    
    for (int i = 0; i < this->v_iCircuit[this->v_iCircuit.size() - 1].size(); i++) {
        float error;
        if (i == correct) {
            error = 0.5 * ((0.01 - this->v_iCircuit[this->v_iCircuit.size() - 1][i]) * (0.01 - this->v_iCircuit[this->v_iCircuit.size() - 1][i]));
        } else {
            error = 0.5 * ((0.99 - this->v_iCircuit[this->v_iCircuit.size() - 1][i]) * (0.99 - this->v_iCircuit[this->v_iCircuit.size() - 1][i]));
        }
        error_total += error;
        error_out.push_back(error);
    }
    
    std::vector<std::vector<std::vector<float>>> et_ox;
    et_ox.push_back(std::vector<std::vector<float>>());
    std::vector<std::vector<std::vector<float>>> nw = this->v_fWeights;
    
    // mechanical delta
    for (int i = 0; i < this->v_iCircuit[this->v_iCircuit.size() - 1].size(); i++) {
        float et_on, on_nn;
        float target = (i == correct) ? 0.99 : 0.01;
        et_on = -(target - this->v_iCircuit[this->v_iCircuit.size() - 1][i]);
        on_nn = (this->e_iFunction == ACTIVATION_SIGMOID) ? brain::DerivativeSigmoid(this->v_iCircuit[this->v_iCircuit.size() - 1][i])
                : (this->e_iFunction == ACTIVATION_TANH) ? brain::DerivativeTanh(this->v_iCircuit[this->v_iCircuit.size() - 1][i])
                : (this->e_iFunction == ACTIVATION_SOFTPLUS) ? brain::DerivativeSoftplus(this->v_iCircuit[this->v_iCircuit.size() - 1][i])
                : brain::DerivativeReLu(this->v_iCircuit[this->v_iCircuit.size() - 1][i]);
        
        float delta_o = et_on * on_nn;
        //delta[this->v_iCircuit.size() - 1][i][0] = delta_o;
        
        for (int n = 0; n < this->v_iCircuit[this->v_iCircuit.size() - 2].size(); n++) {
            float no_wn = this->v_iCircuit[this->v_iCircuit.size() - 2][n];
            float et_wn = delta_o * no_wn;
            float weight_new = this->v_fWeights[this->v_iCircuit.size() - 2][n][i] - (this->f_Eta * et_wn);
            nw[this->v_iCircuit.size() - 2][n][i] = weight_new;
        }
    }
    
    // hidden delta
    for (int i = (int) this->v_iCircuit.size() - 2; i >= 0; i--) {
        float et_on, on_nn;
        
        
    }
    
    /*for (int i = 0; i < this->v_iCircuit[this->v_iCircuit.size() - 1].size(); i++) {
        float error_out;
        
        if (i == correct) {
            error_out = -(0.99 - this->v_iCircuit[this->v_iCircuit.size() - 1][i]) * (this->v_iCircuit[this->v_iCircuit.size() - 1][i] * (1 - this->v_iCircuit[this->v_iCircuit.size() - 1][i]));
        } else {
            error_out = -(0.01 - this->v_iCircuit[this->v_iCircuit.size() - 1][i]) * (this->v_iCircuit[this->v_iCircuit.size() - 1][i] * (1 - this->v_iCircuit[this->v_iCircuit.size() - 1][i]));
        }
        
        // backwards loop through CNN from L
        for (int n = (int) this->v_iCircuit.size() - 1; n >= 0; n--) {
            // loop through current layer
            for (int j = 0; j < this->v_iCircuit[n].size(); j++) {
                float delta, nw;
                
                if (n == this->v_iCircuit.size() - 1) {
                    // for mechanical layer
                    delta = error_out * this->v_fWeights[n-1][j][i];
                    nw = this->v_fWeights[n-1][j][i] - (this->f_Eta * delta);
                } else {
                    // for deep/sensory layers
                    // calculate delta looking at all neurons it affects
                    delta = 0;
                    
                    for (int k = 0; k < this->v_fWeights[n][j].size(); k++) {
                        
                    }
                }
            }
        }
    }*/
}
