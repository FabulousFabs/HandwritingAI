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

double brain::MakeRandomP() {
    double r = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
    return r;
}

double brain::MakeRandomN() {
    double r = MakeRandomP() * (-1);
    return r;
}

double brain::MakeRandomNP() {
    double r1 = MakeRandomP(), r2 = MakeRandomP();
    
    if (r2 < 0.5) {
        r1 = r1 * (-1);
    }
    
    return r1;
}

double brain::MakeRandomXavier(int ins, int outs) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    double variance = 2.0 / (double) (ins + outs);
    double stdev = sqrt(variance);
    
    std::normal_distribution<double> p((double) 0, stdev);
    return p(gen);
}

//
//
// brain::activation & brain::derivation functions

std::vector<std::vector<double>> brain::ActivationSigmoid(std::vector<std::vector<double>> ins) {
    for (int i = 0; i < ins.size(); i++) {
        for (int j = 0; j < ins[i].size(); j++) {
            ins[i][j] = ActivationSigmoid(ins[i][j]);
        }
    }
    return ins;
}

double brain::ActivationSigmoid(double in) {
    double out = 1 / (1 + exp(-in));
    return out;
}

std::vector<std::vector<double>> brain::DerivativeSigmoid(std::vector<std::vector<double>> ins) {
    for (int i = 0; i < ins.size(); i++) {
        for (int j = 0; j < ins[i].size(); j++) {
            ins[i][j] = DerivativeSigmoid(ins[i][j]);
        }
    }
    return ins;
}

double brain::DerivativeSigmoid(double in) {
    double out = ActivationSigmoid(in) * (1 - ActivationSigmoid(in));
    return out;
}

std::vector<std::vector<double>> brain::ActivationTanh(std::vector<std::vector<double>> ins) {
    for (int i = 0; i < ins.size(); i++) {
        for (int j = 0; j < ins[i].size(); j++) {
            ins[i][j] = ActivationTanh(ins[i][j]);
        }
    }
    return ins;
}

double brain::ActivationTanh(double in) {
    double out = (2 / (1 + exp(-2 * in))) - 1;
    return out;
}

std::vector<std::vector<double>> brain::DerivativeTanh(std::vector<std::vector<double>> ins) {
    for (int i = 0; i < ins.size(); i++) {
        for (int j = 0; j < ins[i].size(); j++) {
            ins[i][j] = DerivativeTanh(ins[i][j]);
        }
    }
    return ins;
}

double brain::DerivativeTanh(double in) {
    double out = 1 - (ActivationTanh(in) * ActivationTanh(in));
    return out;
}

std::vector<std::vector<double>> brain::ActivationSoftplus(std::vector<std::vector<double>> ins) {
    for (int i = 0; i < ins.size(); i++) {
        for (int j = 0; j < ins[i].size(); j++) {
            ins[i][j] = ActivationSoftplus(ins[i][j]);
        }
    }
    return ins;
}

double brain::ActivationSoftplus(double in) {
    double out = log(1 + exp(in));
    return out;
}

std::vector<std::vector<double>> brain::DerivativeSoftplus(std::vector<std::vector<double>> ins) {
    for (int i = 0; i < ins.size(); i++) {
        for (int j = 0; j < ins[i].size(); j++) {
            ins[i][j] = DerivativeSoftplus(ins[i][j]);
        }
    }
    return ins;
}

double brain::DerivativeSoftplus(double in) {
    double out = 1 / (1 + exp(-in));
    return out;
}

std::vector<std::vector<double>> brain::ActivationReLu(std::vector<std::vector<double>> ins) {
    for (int i = 0; i < ins.size(); i++) {
        for (int j = 0; j < ins[i].size(); j++) {
            ins[i][j] = ActivationReLu(ins[i][j]);
        }
    }
    return ins;
}

double brain::ActivationReLu(double in) {
    double out = std::max((double) 0, in);
    return out;
}

std::vector<std::vector<double>> brain::DerivativeReLu(std::vector<std::vector<double>> ins) {
    for (int i = 0; i < ins.size(); i++) {
        for (int j = 0; j < ins[i].size(); j++) {
            ins[i][j] = DerivativeReLu(ins[i][j]);
        }
    }
    return ins;
}

double brain::DerivativeReLu(double in) {
    double out = (in <= 0) ? (double) 0 : (double) 1;
    return out;
}

std::vector<std::vector<double>> brain::ActivationELU(std::vector<std::vector<double>> ins) {
    for (int i = 0; i < ins.size(); i++) {
        for (int j = 0; j < ins[i].size(); j++) {
            ins[i][j] = ActivationELU(ins[i][j]);
        }
    }
    return ins;
}

double brain::ActivationELU(double in) {
    double out = std::max((double) 0, in);
    if (out == 0) {
        out = 1 * (exp(in) - 1);
    }
    return out;
}

std::vector<std::vector<double>> brain::DerivativeELU(std::vector<std::vector<double>> ins) {
    for (int i = 0; i < ins.size(); i++) {
        for (int j = 0; j < ins[i].size(); j++) {
            ins[i][j] = DerivativeELU(ins[i][j]);
        }
    }
    return ins;
}

double brain::DerivativeELU(double in) {
    double out = (in <= 0) ? (double) 0 : (double) 1;
    if (out == 0) {
        out = 1 * (exp(in));
    }
    return out;
}

std::vector<std::vector<double>> brain::ActivationReLuLeaky(std::vector<std::vector<double>> ins) {
    for (int i = 0; i < ins.size(); i++) {
        for (int j = 0; j < ins[i].size(); j++) {
            ins[i][j] = ActivationReLuLeaky(ins[i][j]);
        }
    }
    return ins;
}

double brain::ActivationReLuLeaky(double in) {
    double out = std::max((double) 0, in);
    if (out == 0) {
        out = 0.01 * in;
    }
    return out;
}

std::vector<std::vector<double>> brain::DerivativeReLuLeaky(std::vector<std::vector<double>> ins) {
    for (int i = 0; i < ins.size(); i++) {
        for (int j = 0; j < ins[i].size(); j++) {
            ins[i][j] = DerivativeReLuLeaky(ins[i][j]);
        }
    }
    return ins;
}

double brain::DerivativeReLuLeaky(double in) {
    double out = (in <= 0) ? (double) 0.01 : (double) 1;
    return out;
}

std::vector<std::vector<double>> brain::Activate(std::vector<std::vector<double>> ins, enum ACTIVATION_FUNC af) {
    for (int i = 0; i < ins.size(); i++) {
        for (int j = 0; j < ins[i].size(); j++) {
            ins[i][j] = brain::Activate(ins[i][j], af);
        }
    }
    return ins;
}

double brain::Activate(double in, enum ACTIVATION_FUNC af) {
    double out = (af == ACTIVATION_SIGMOID) ? ActivationSigmoid(in)
                : (af == ACTIVATION_TANH) ? ActivationTanh(in)
                : (af == ACTIVATION_SOFTPLUS) ? ActivationSoftplus(in)
                : (af == ACTIVATION_RELU) ? ActivationReLu(in)
                : (af == ACTIVATION_ELU) ? ActivationELU(in)
                : ActivationReLuLeaky(in);
    return out;
}

std::vector<std::vector<double>> brain::Derive(std::vector<std::vector<double>> ins, enum ACTIVATION_FUNC af) {
    for (int i = 0; i < ins.size(); i++) {
        for (int j = 0; j < ins[i].size(); j++) {
            ins[i][j] = brain::Derive(ins[i][j], af);
        }
    }
    return ins;
}

double brain::Derive(double in, enum ACTIVATION_FUNC af) {
    double out = (af == ACTIVATION_SIGMOID) ? DerivativeSigmoid(in)
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

double brain::LossSparseCategoricalCrossEntropy(double obs, double exp) {
    obs = (obs >= 0.9999) ? 0.9999 : obs; // in case output is SOFTPLUS
    double out = (exp == 1) ? -log(abs(obs)) * ((obs < 0) ? -1 : 1) : -log(((double) 1-obs));
    return out;
}

double brain::LossSquare(double obs, double exp) {
    double out = (exp - obs) * (exp - obs);
    return out;
}

double brain::Loss(int percept, double p, int correct, enum brain::optimiser::LOSS_FUNC lf) {
    double out = (lf == optimiser::LOSS_MEAN_SQUARE || lf == optimiser::LOSS_SQUARE) ? LossSquare(p, (percept == correct) ? 0.99 : 0.01)
            : LossSparseCategoricalCrossEntropy(p, (percept == correct) ? 1 : 0);
    
    return out;
}

std::vector<double> brain::Loss(std::vector<double> &outs, int correct, enum brain::optimiser::LOSS_FUNC lf) {
    std::vector<double> v;
    
    for (int i = 0; i < outs.size(); i++) {
        v.push_back(Loss(i, outs[i], correct, lf));
    }
    
    double s = 0;
    for (int i = 0; i < v.size(); i++) {
        s = s + v[i];
    }
    
    for (int i = 0; i < v.size(); i++) {
        v[i] = (lf == optimiser::LOSS_MEAN_SQUARE) ? s / v.size()
            : v[i];
    }
    
    return v;
}

double brain::DeriveLossSquare(double in) {
    double out = 2 * in;
    return out;
}

double brain::DeriveLossSparseCategoricalCrossEntropy(double in) {
    double out = (1 / in);
    return out;
}

double brain::DeriveLoss(double in, enum brain::optimiser::LOSS_FUNC lf) {
    double out = (lf == optimiser::LOSS_SQUARE || lf == optimiser::LOSS_MEAN_SQUARE) ? brain::DeriveLossSquare(in)
                : brain::DeriveLossSparseCategoricalCrossEntropy(in);
    return out;
}

std::vector<double> brain::DeriveLoss(std::vector<double> &ins, enum brain::optimiser::LOSS_FUNC lf) {
    std::vector<double> outs;
    
    for (auto&& f:ins) {
        outs.push_back(DeriveLoss(f, lf));
    }
    
    return outs;
}

//
//
// brain::matrix functions

std::vector<std::vector<double>> brain::MatrixDot(std::vector<std::vector<double>> &m1, std::vector<std::vector<double>> &m2) {
    assert(m1.size() == m2[0].size());
    assert(m1[0].size() == m2.size());
    
    std::vector<std::vector<double>> product;
    for (int i = 0; i < m1.size(); i++) {
        std::vector<double> t;
        for (int j = 0; j < m2[0].size(); j++) {
            t.push_back((double) 0);
        }
        product.push_back(t);
    }
    
    for (int i = 0; i < m1.size(); i++) {
        for (int j = 0; j < m2[0].size(); j++) {
            for (int k = 0; k < m1[0].size(); k++) {
                product[i][j] = product[i][j] + (m1[i][k] * m2[k][j]);
            }
        }
    }
    
    return product;
}

void brain::MatrixFill(bool r, double f, int u, std::vector<std::vector<double>> &m) {
    if (r) {
        std::vector<double> t;
        
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

void brain::MatrixOnes(std::vector<std::vector<double>> &m1, std::vector<std::vector<double>> &m2) {
    if (m1.size() != m2.size()) {
        if (m1.size() > m2.size()) {
            MatrixFill(true, (double) 1, (int) m1.size(), m2);
        } else {
            MatrixFill(true, (double) 1, (int) m2.size(), m1);
        }
    } else {
        for (int i = 0; i < m1.size(); i++) {
            if (m1[i].size() != m2[i].size()) {
                if (m1[i].size() > m2[i].size()) {
                    MatrixFill(false, (double) 1, (int) m1[i].size(), m2);
                } else {
                    MatrixFill(false, (double) 1, (int) m2[i].size(), m1);
                }
            }
        }
    }
}

void brain::MatrixZeroes(std::vector<std::vector<double>> &m1, std::vector<std::vector<double>> &m2) {
    if (m1.size() != m2.size()) {
        if (m1.size() > m2.size()) {
            MatrixFill(true, (double) 0, (int) m1.size(), m2);
        } else {
            MatrixFill(true, (double) 0, (int) m2.size(), m1);
        }
    } else {
        for (int i = 0; i < m1.size(); i++) {
            if (m1[i].size() != m2[i].size()) {
                if (m1[i].size() > m2[i].size()) {
                    MatrixFill(false, (double) 0, (int) m1[i].size(), m2);
                } else {
                    MatrixFill(false, (double) 0, (int) m2[i].size(), m1);
                }
            }
        }
    }
}

void brain::MatrixFit(std::vector<std::vector<double>> &m1, std::vector<std::vector<double>> &m2) {
    while (m1.size() != m2.size()) {
        MatrixOnes(m1, m2);
    }
    
    for (int i = 0; i < m1.size(); i++) {
        while (m1[i].size() != m2[i].size()) {
            MatrixOnes(m1, m2);
        }
    }
}

std::vector<std::vector<double>> brain::MatrixT(std::vector<std::vector<double>> &m) {
    std::vector<std::vector<double>> v;
    
    for (int i = 0; i < m[0].size(); i++){
        std::vector<double> v1;
        
        for (int j = 0; j < m.size(); j++) {
            v1.push_back(m[j][i]);
        }
        
        v.push_back(v1);
    }
    
    return v;
}
