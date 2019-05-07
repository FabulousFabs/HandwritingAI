//
//  brain.cpp
//  CppMLPHandwriting
//
//  Created by Fabian Schneider on 24.04.19.
//  Copyright © 2019 Fabian Schneider. All rights reserved.
//

#include "brain.hpp"

using namespace brain;

//
//
// brain::random functions

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

float brain::MakeRandomXavier(int ins, int outs) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    float variance = 2.0 / (float) (ins + outs);
    float stdev = sqrt(variance);
    
    std::normal_distribution<float> p((float) 0, stdev);
    return p(gen);
}

//
//
// brain::activation & brain::derivation functions

std::vector<std::vector<float>> brain::ActivationSigmoid(std::vector<std::vector<float>> ins) {
    for (int i = 0; i < ins.size(); i++) {
        for (int j = 0; j < ins[i].size(); j++) {
            ins[i][j] = ActivationSigmoid(ins[i][j]);
        }
    }
    return ins;
}

float brain::ActivationSigmoid(float in) {
    float out = 1 / (1 + exp(-in));
    return out;
}

std::vector<std::vector<float>> brain::DerivativeSigmoid(std::vector<std::vector<float>> ins) {
    for (int i = 0; i < ins.size(); i++) {
        for (int j = 0; j < ins[i].size(); j++) {
            ins[i][j] = DerivativeSigmoid(ins[i][j]);
        }
    }
    return ins;
}

float brain::DerivativeSigmoid(float in) {
    float out = ActivationSigmoid(in) * (1 - ActivationSigmoid(in));
    return out;
}

std::vector<std::vector<float>> brain::ActivationTanh(std::vector<std::vector<float>> ins) {
    for (int i = 0; i < ins.size(); i++) {
        for (int j = 0; j < ins[i].size(); j++) {
            ins[i][j] = ActivationTanh(ins[i][j]);
        }
    }
    return ins;
}

float brain::ActivationTanh(float in) {
    float out = (2 / (1 + exp(-2 * in))) - 1;
    return out;
}

std::vector<std::vector<float>> brain::DerivativeTanh(std::vector<std::vector<float>> ins) {
    for (int i = 0; i < ins.size(); i++) {
        for (int j = 0; j < ins[i].size(); j++) {
            ins[i][j] = DerivativeTanh(ins[i][j]);
        }
    }
    return ins;
}

float brain::DerivativeTanh(float in) {
    float out = 1 - (ActivationTanh(in) * ActivationTanh(in));
    return out;
}

std::vector<std::vector<float>> brain::ActivationSoftplus(std::vector<std::vector<float>> ins) {
    for (int i = 0; i < ins.size(); i++) {
        for (int j = 0; j < ins[i].size(); j++) {
            ins[i][j] = ActivationSoftplus(ins[i][j]);
        }
    }
    return ins;
}

float brain::ActivationSoftplus(float in) {
    float out = log(1 + exp(in));
    return out;
}

std::vector<std::vector<float>> brain::DerivativeSoftplus(std::vector<std::vector<float>> ins) {
    for (int i = 0; i < ins.size(); i++) {
        for (int j = 0; j < ins[i].size(); j++) {
            ins[i][j] = DerivativeSoftplus(ins[i][j]);
        }
    }
    return ins;
}

float brain::DerivativeSoftplus(float in) {
    float out = 1 / (1 + exp(-in));
    return out;
}

std::vector<std::vector<float>> brain::ActivationReLu(std::vector<std::vector<float>> ins) {
    for (int i = 0; i < ins.size(); i++) {
        for (int j = 0; j < ins[i].size(); j++) {
            ins[i][j] = ActivationReLu(ins[i][j]);
        }
    }
    return ins;
}

float brain::ActivationReLu(float in) {
    float out = std::max((float) 0, in);
    return out;
}

std::vector<std::vector<float>> brain::DerivativeReLu(std::vector<std::vector<float>> ins) {
    for (int i = 0; i < ins.size(); i++) {
        for (int j = 0; j < ins[i].size(); j++) {
            ins[i][j] = DerivativeReLu(ins[i][j]);
        }
    }
    return ins;
}

float brain::DerivativeReLu(float in) {
    float out = (in <= 0) ? (float) 0 : (float) 1;
    return out;
}

std::vector<std::vector<float>> brain::ActivationELU(std::vector<std::vector<float>> ins) {
    for (int i = 0; i < ins.size(); i++) {
        for (int j = 0; j < ins[i].size(); j++) {
            ins[i][j] = ActivationELU(ins[i][j]);
        }
    }
    return ins;
}

float brain::ActivationELU(float in) {
    float out = std::max((float) 0, in);
    if (out == 0) {
        out = 1 * (exp(in) - 1);
    }
    return out;
}

std::vector<std::vector<float>> brain::DerivativeELU(std::vector<std::vector<float>> ins) {
    for (int i = 0; i < ins.size(); i++) {
        for (int j = 0; j < ins[i].size(); j++) {
            ins[i][j] = DerivativeELU(ins[i][j]);
        }
    }
    return ins;
}

float brain::DerivativeELU(float in) {
    float out = (in <= 0) ? (float) 0 : (float) 1;
    if (out == 0) {
        out = 1 * (exp(in));
    }
    return out;
}

std::vector<std::vector<float>> brain::ActivationReLuLeaky(std::vector<std::vector<float>> ins) {
    for (int i = 0; i < ins.size(); i++) {
        for (int j = 0; j < ins[i].size(); j++) {
            ins[i][j] = ActivationReLuLeaky(ins[i][j]);
        }
    }
    return ins;
}

float brain::ActivationReLuLeaky(float in) {
    float out = std::max((float) 0, in);
    if (out == 0) {
        out = 0.01 * in;
    }
    return out;
}

std::vector<std::vector<float>> brain::DerivativeReLuLeaky(std::vector<std::vector<float>> ins) {
    for (int i = 0; i < ins.size(); i++) {
        for (int j = 0; j < ins[i].size(); j++) {
            ins[i][j] = DerivativeReLuLeaky(ins[i][j]);
        }
    }
    return ins;
}

float brain::DerivativeReLuLeaky(float in) {
    float out = (in <= 0) ? (float) 0.01 : (float) 1;
    return out;
}

std::vector<std::vector<float>> brain::Activate(std::vector<std::vector<float>> ins, enum ACTIVATION_FUNC af) {
    for (int i = 0; i < ins.size(); i++) {
        for (int j = 0; j < ins[i].size(); j++) {
            ins[i][j] = brain::Activate(ins[i][j], af);
        }
    }
    return ins;
}

float brain::Activate(float in, enum ACTIVATION_FUNC af) {
    float out = (af == ACTIVATION_SIGMOID) ? ActivationSigmoid(in)
                : (af == ACTIVATION_TANH) ? ActivationTanh(in)
                : (af == ACTIVATION_SOFTPLUS) ? ActivationSoftplus(in)
                : (af == ACTIVATION_RELU) ? ActivationReLu(in)
                : (af == ACTIVATION_ELU) ? ActivationELU(in)
                : ActivationReLuLeaky(in);
    return out;
}

std::vector<std::vector<float>> brain::Derive(std::vector<std::vector<float>> ins, enum ACTIVATION_FUNC af) {
    for (int i = 0; i < ins.size(); i++) {
        for (int j = 0; j < ins[i].size(); j++) {
            ins[i][j] = brain::Derive(ins[i][j], af);
        }
    }
    return ins;
}

float brain::Derive(float in, enum ACTIVATION_FUNC af) {
    float out = (af == ACTIVATION_SIGMOID) ? DerivativeSigmoid(in)
                : (af == ACTIVATION_TANH) ? DerivativeTanh(in)
                : (af == ACTIVATION_SOFTPLUS) ? DerivativeSoftplus(in)
                : (af == ACTIVATION_RELU) ? DerivativeReLu(in)
                : (af == ACTIVATION_ELU) ? DerivativeELU(in)
                : DerivativeReLuLeaky(in);
    return out;
}

//
//
// brain::loss functions

float brain::LossSparseCategoricalCrossEntropy(float obs, float exp) {
    obs = (obs >= 1) ? 0.99 : obs; // in case output is SOFTPLUS
    float out = (exp == 1) ? -log(obs) : -log(1 - obs);
    return out;
}

float brain::LossSquare(float obs, float exp) {
    float out = (exp - obs) * (exp - obs);
    return out;
}

float brain::Loss(int percept, float p, int correct, enum brain::optimiser::LOSS_FUNC lf) {
    float out = (lf == optimiser::LOSS_MEAN_SQUARE || lf == optimiser::LOSS_SQUARE) ? LossSquare(p, (percept == correct) ? 0.99 : 0.01)
            : LossSparseCategoricalCrossEntropy(p, (percept == correct) ? 1 : 0);
    return out;
}

std::vector<float> brain::Loss(std::vector<float> &outs, int correct, enum brain::optimiser::LOSS_FUNC lf) {
    std::vector<float> v;
    
    for (int i = 0; i < outs.size(); i++) {
        v.push_back(Loss(i, outs[i], correct, lf));
    }
    
    float s = 0;
    for (int i = 0; i < v.size(); i++) {
        s += v[i];
    }
    
    for (int i = 0; i < v.size(); i++) {
        v[i] = (lf == optimiser::LOSS_MEAN_SQUARE) ? s / v.size()
            : v[i];
    }
    
    return v;
}

//
//
// brain::matrix functions

std::vector<std::vector<float>> brain::MatrixDot(std::vector<std::vector<float>> &m1, std::vector<std::vector<float>> &m2) {
    assert(m1.size() == m2[0].size());
    assert(m1[0].size() == m2.size());
    
    std::vector<std::vector<float>> product;
    for (int i = 0; i < m1.size(); i++) {
        std::vector<float> t;
        for (int j = 0; j < m2[0].size(); j++) {
            t.push_back((float) 0);
        }
        product.push_back(t);
    }
    
    for (int i = 0; i < m1.size(); i++) {
        for (int j = 0; j < m2[0].size(); j++) {
            for (int k = 0; k < m1[0].size(); k++) {
                product[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }
    
    return product;
}

void brain::MatrixFill(bool r, float f, int u, std::vector<std::vector<float>> &m) {
    if (r) {
        std::vector<float> t;
        
        for (int i = 0; i < (int) m[0].size(); i++) {
            t.push_back(f);
        }
        
        for (int i = (int) m.size(); i < u; i++) {
            m.push_back(t);
        }
    } else {
        for (int i = 0; i < m.size(); i++) {
            for (int j = (int) m[i].size(); j < u; j++) {
                m[i].push_back(f);
            }
        }
    }
}

void brain::MatrixOnes(std::vector<std::vector<float>> &m1, std::vector<std::vector<float>> &m2) {
    if (m1.size() != m2.size()) {
        if (m1.size() > m2.size()) {
            MatrixFill(true, (float) 1, (int) m1.size(), m2);
        } else {
            MatrixFill(true, (float) 1, (int) m2.size(), m1);
        }
    } else {
        for (int i = 0; i < m1.size(); i++) {
            if (m1[i].size() != m2[i].size()) {
                if (m1[i].size() > m2[i].size()) {
                    MatrixFill(false, (float) 1, (int) m1[i].size(), m2);
                } else {
                    MatrixFill(false, (float) 1, (int) m2[i].size(), m1);
                }
            }
        }
    }
}

void brain::MatrixZeroes(std::vector<std::vector<float>> &m1, std::vector<std::vector<float>> &m2) {
    if (m1.size() != m2.size()) {
        if (m1.size() > m2.size()) {
            MatrixFill(true, (float) 0, (int) m1.size(), m2);
        } else {
            MatrixFill(true, (float) 0, (int) m2.size(), m1);
        }
    } else {
        for (int i = 0; i < m1.size(); i++) {
            if (m1[i].size() != m2[i].size()) {
                if (m1[i].size() > m2[i].size()) {
                    MatrixFill(false, (float) 0, (int) m1[i].size(), m2);
                } else {
                    MatrixFill(false, (float) 0, (int) m2[i].size(), m1);
                }
            }
        }
    }
}

void brain::MatrixFit(std::vector<std::vector<float>> &m1, std::vector<std::vector<float>> &m2) {
    while (m1.size() != m2.size()) {
        MatrixOnes(m1, m2);
    }
    
    for (int i = 0; i < m1.size(); i++) {
        while (m1[i].size() != m2[i].size()) {
            MatrixOnes(m1, m2);
        }
    }
}

std::vector<std::vector<float>> brain::MatrixT(std::vector<std::vector<float>> &m) {
    std::vector<std::vector<float>> v;
    
    for (int i = 0; i < m[0].size(); i++){
        std::vector<float> v1;
        
        for (int j = 0; j < m.size(); j++) {
            v1.push_back(m[j][i]);
        }
        
        v.push_back(v1);
    }
    
    return v;
}
