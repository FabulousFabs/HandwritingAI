//
//  brain.cpp
//  CppCNNHandwriting
//
//  Created by Fabian Schneider on 24.04.19.
//  Copyright © 2019 Fabian Schneider. All rights reserved.
//

#include "brain.hpp"

using namespace brain;

/*std::vector<int> brain::MakeCircuitVector(int n_args, ...) {
    std::vector<int> v;

    va_list ap;
    va_start(ap, n_args);
        
    for (int i = 1; i <= n_args; i++) {
        int a = va_arg(ap, int);
        v.push_back(a);
    }
    
    va_end(ap);
    
    return v;
}*/

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

float brain::Activate(float in, enum ACTIVATION_FUNC af) {
    float out = (af == ACTIVATION_SIGMOID) ? ActivationSigmoid(in)
                : (af == ACTIVATION_TANH) ? ActivationTanh(in)
                : (af == ACTIVATION_SOFTPLUS) ? ActivationSoftplus(in)
                : ActivationReLu(in);
    return out;
}

float brain::Derive(float in, enum ACTIVATION_FUNC af) {
    float out = (af == ACTIVATION_SIGMOID) ? DerivativeSigmoid(in)
                : (af == ACTIVATION_TANH) ? DerivativeTanh(in)
                : (af == ACTIVATION_SOFTPLUS) ? DerivativeSoftplus(in)
                : DerivativeReLu(in);
    return out;
}

std::vector<std::vector<float>> brain::MatrixDot(std::vector<std::vector<float>> m1, std::vector<std::vector<float>> m2) {
    assert(m1.size() == m2[0].size());
    assert(m1[0].size() == m2.size());
    
    std::vector<std::vector<float>> product;
    for (int i = 0; i < m1.size(); i++) {
        std::vector<float> t;
        for (int j = 0; j < m1[i].size(); j++) {
            t.push_back((float) 0);
        }
        product.push_back(t);
    }
    
    for (int i = 0; i < m1.size(); i++) {
        for (int j = 0; j < m1[i].size(); j++) {
            for (int k = 0; k < m2[i].size(); k++) {
                product[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }
    
    /*for (int i = 0; i <= m1.size() - 1; i++) {
        std::vector<float> t;
        for (int n = 0; n <= m1[i].size() - 1; n++) {
            float p = m1[i][n] * m2[n][i];
            t.push_back(p);
        }
        product.push_back(t);
    }*/
    
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
