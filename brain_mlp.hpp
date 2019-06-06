//
//  brain_MLP.hpp
//  CppMLPHandwriting
//
//  Created by Fabian Schneider on 30.04.19.
//  Copyright © 2019 Fabian Schneider. All rights reserved.
//

#ifndef brain_MLP_hpp
#define brain_MLP_hpp

#include <iostream>
#include <vector>
#include <stdio.h>
#include <algorithm>
#include <stdarg.h>
#include <tuple>

#include "brain.hpp"
#include "main.h"

namespace brain {
    enum MLP_LAYER_TYPE
    {
        MLP_LAYER_T_FLATTEN = 0,
        MLP_LAYER_T_DENSE,
    };
    
    class layer_proto
    {
    protected:
        int i_nNeurons;
        int i_nWeights;
    public:
        enum MLP_LAYER_TYPE e_iType;
        enum ACTIVATION_FUNC e_iActivationFunc;
        std::vector<double> v_fNeurons;
        std::vector<double> v_fNeuronsOut;
        std::vector<std::vector<double>> v_fWeights;
        
        void Grow();
        void Neuroplasticity(enum brain::WEIGHTS_INIT wi, int ins, int outs);
        void GetSensations(std::vector<double> s);
        void Excite(brain::layer_proto *next_layer);
        void Activate();
    };
    
    namespace layer {
        class flatten_proto: public layer_proto
        {
        private:
            
        public:
            flatten_proto(int n, enum ACTIVATION_FUNC af);
            void Excite(brain::layer_proto *next_layer);
        };
        
        class dense_proto: public layer_proto
        {
        private:
            
        public:
            dense_proto(int n, enum ACTIVATION_FUNC af);
        };
        
        std::tuple<enum MLP_LAYER_TYPE, int, enum ACTIVATION_FUNC> Flatten(int n_args, ...);
        std::tuple<enum MLP_LAYER_TYPE, int, enum ACTIVATION_FUNC> Dense(int neurons, enum ACTIVATION_FUNC af);
    }
    
    
    
    class MLP
    {
    private:
        std::vector<layer_proto> Layers;
        bool b_IsCompiled;
        double f_LearningRate;
        enum brain::optimiser::OPTIMISER_TYPE e_iOptimiser;
        enum brain::optimiser::LOSS_FUNC e_iLossFunc;
        void StochasticGradientDescentOptimisation(int percept, int correct);
    public:
        MLP();
        void Sequential(std::tuple<enum brain::MLP_LAYER_TYPE, int, enum brain::ACTIVATION_FUNC> l);
        void Compile(enum brain::WEIGHTS_INIT wi, enum brain::optimiser::OPTIMISER_TYPE ot, enum brain::optimiser::LOSS_FUNC lf, double lr);
        void Flush();
        std::tuple<int, double> Perceive(std::vector<double> &s);
        std::tuple<int, double> GetChoice();
        void Feedback(int correct);
        std::tuple<int, double> Train(std::vector<double> &s, int correct);
    };
}

#endif /* brain_MLP_hpp */
