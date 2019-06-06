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
    double MakeRandomP();
    double MakeRandomN();
    double MakeRandomNP();
    double MakeRandomXavier(int ins, int outs);
    
    enum WEIGHTS_INIT
    {
        WEIGHTS_INIT_RANDOM = 0,
        WEIGHTS_INIT_XAVIER,
        WEIGHTS_INIT_POINT_O_ONE
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
    
    namespace optimiser {
        enum OPTIMISER_TYPE
        {
            OPTIMISER_ADAM = 0,
            OPTIMISER_SGD
        };
        
        enum LOSS_FUNC
        {
            LOSS_SPARSE_CATEGORICAL_CROSSENTROPY = 0,
            LOSS_MEAN_SQUARE,
            LOSS_SQUARE
        };
    }
    
    std::vector<std::vector<double>> ActivationSigmoid(std::vector<std::vector<double>> ins);
    double ActivationSigmoid(double in);
    std::vector<std::vector<double>> DerivativeSigmoid(std::vector<std::vector<double>> ins);
    double DerivativeSigmoid(double in);
    
    double ActivationTanh(double in);
    std::vector<std::vector<double>> ActivationTanh(std::vector<std::vector<double>> ins);
    double DerivativeTanh(double in);
    std::vector<std::vector<double>> DerivativeTanh(std::vector<std::vector<double>> ins);
    
    double ActivationSoftplus(double in);
    std::vector<std::vector<double>> ActivationSoftplus(std::vector<std::vector<double>> ins);
    double DerivativeSoftplus(double in);
    std::vector<std::vector<double>> DerivativeSoftplus(std::vector<std::vector<double>> ins);
    
    double ActivationReLu(double in);
    std::vector<std::vector<double>> ActivationReLu(std::vector<std::vector<double>> ins);
    double DerivativeReLu(double in);
    std::vector<std::vector<double>> DerivativeReLu(std::vector<std::vector<double>> ins);
    
    double ActivationELU(double in);
    std::vector<std::vector<double>> ActivationELU(std::vector<std::vector<double>> ins);
    double DerivativeELU(double in);
    std::vector<std::vector<double>> DerivativeELU(std::vector<std::vector<double>> ins);
    
    double ActivationReLuLeaky(double in);
    std::vector<std::vector<double>> ActivationReLuLeaky(std::vector<std::vector<double>> ins);
    double DerivativeReLuLeaky(double in);
    std::vector<std::vector<double>> DerivativeReLuLeaky(std::vector<std::vector<double>> ins);
    
    double Activate(double in, enum ACTIVATION_FUNC af);
    std::vector<std::vector<double>> Activate(std::vector<std::vector<double>> ins, enum ACTIVATION_FUNC af);
    double Derive(double in, enum ACTIVATION_FUNC af);
    std::vector<std::vector<double>> Derive(std::vector<std::vector<double>> ins, enum ACTIVATION_FUNC af);
    
    double LossSquare(double obs, double exp);
    double DeriveLossSquare(double in);
    double LossSparseCategoricalCrossEntropy(double obs, double exp);
    double DeriveLossSparseCategoricalCrossEntropy(double in);
    
    double Loss(int percept, double p, int correct, enum optimiser::LOSS_FUNC lf);
    std::vector<double> Loss(std::vector<double> &outs, int correct, enum optimiser::LOSS_FUNC lf);
    
    double DeriveLoss(double in, enum optimiser::LOSS_FUNC lf);
    std::vector<double> DeriveLoss(std::vector<double> &ins, enum optimiser::LOSS_FUNC lf);
    
    std::vector<std::vector<double>> MatrixDot(std::vector<std::vector<double>> &m1, std::vector<std::vector<double>> &m2);
    void MatrixFill(bool r, double f, int u, std::vector<std::vector<double>> &m);
    void MatrixOnes(std::vector<std::vector<double>> &m1, std::vector<std::vector<double>> &m2);
    void MatrixZeroes(std::vector<std::vector<double>> &m1, std::vector<std::vector<double>> &m2);
    void MatrixFit(std::vector<std::vector<double>> &m1, std::vector<std::vector<double>> &m2);
    std::vector<std::vector<double>> MatrixT(std::vector<std::vector<double>> &m);
}

#include "brain_mlp.hpp"
#include "brain_pne.hpp"

#endif /* brain_hpp */
