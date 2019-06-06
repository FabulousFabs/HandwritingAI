//
//  brain_pne.hpp
//  CppCNNHandwriting
//
//  Created by Fabian Schneider on 05.05.19.
//  Copyright © 2019 Fabian Schneider. All rights reserved.
//

#ifndef brain_pne_hpp
#define brain_pne_hpp

#include <stdio.h>
#include <tuple>
#include <vector>
#include <random>
#include <algorithm>

#include "brain.hpp"

namespace brain {
    enum PNE_LAYER_TYPE
    {
        PNE_LAYER_T_FLATTEN = 0,
        PNE_LAYER_T_CONVOLUTION,
        PNE_LAYER_T_DENSE
    };
    
    class Neuron
    {
        
    };
    
    class PNE
    {
    private:
        std::vector<int> v_iLayerStruct;
        std::vector<Neuron> v_nNeurons;
        bool b_IsCompiled;
    public:
        PNE();
        void Sequential(std::tuple<enum brain::PNE_LAYER_TYPE, int, enum brain::ACTIVATION_FUNC>);
    };
}

#endif /* brain_pne_hpp */
