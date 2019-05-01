//
//  brain_cnn.hpp
//  CppCNNHandwriting
//
//  Created by Fabian Schneider on 30.04.19.
//  Copyright © 2019 Fabian Schneider. All rights reserved.
//

#ifndef brain_cnn_hpp
#define brain_cnn_hpp

#include <iostream>
#include <vector>
#include <stdio.h>
#include <algorithm>
#include <stdarg.h>
#include <tuple>

#include "brain.hpp"

namespace brain {
    enum CNN_LAYER_TYPE
    {
        CNN_LAYER_T_FLATTEN = 0,
        CNN_LAYER_T_DENSE,
    };
    
    class layer_proto
    {
    protected:
        enum CNN_LAYER_TYPE e_iType;
        enum ACTIVATION_FUNC e_iActivationFunc;
        int i_nNeurons;
        int i_nWeights;
    public:
        std::vector<float> v_fNeurons;
        std::vector<float> v_fNeuronsOut;
        std::vector<std::vector<float>> v_fWeights;
        
        void Grow();
        void Neuroplasticity(enum brain::WEIGHTS_INIT wi, int ins, int outs);
        void GetSensations(std::vector<float> s);
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
        
        std::tuple<enum CNN_LAYER_TYPE, int, enum ACTIVATION_FUNC> Flatten(int n_args, ...);
        std::tuple<enum CNN_LAYER_TYPE, int, enum ACTIVATION_FUNC> Dense(int neurons, enum ACTIVATION_FUNC af);
    }
    
    class CNN
    {
    private:
        std::vector<layer_proto> Layers;
        bool b_IsCompiled;
    public:
        CNN();
        void Sequential(std::tuple<enum brain::CNN_LAYER_TYPE, int, enum brain::ACTIVATION_FUNC> l);
        void Compile(enum brain::WEIGHTS_INIT wi);
        std::tuple<int, float> Perceive(std::vector<float> &s);
        std::tuple<int, float> GetChoice();
    };
}

#endif /* brain_cnn_hpp */
