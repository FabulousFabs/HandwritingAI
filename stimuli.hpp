//
//  stimuli.hpp
//  CppCNNHandwriting
//
//  Created by Fabian Schneider on 23.04.19.
//  Copyright © 2019 Fabian Schneider. All rights reserved.
//

#ifndef stimuli_hpp
#define stimuli_hpp

#include <iostream>
#include <vector>
#include <stdlib.h>

#include "filesys.hpp"

#define cimg_display 0
#define cimg_use_png
#include "CImg.h"

namespace stimuli {
    struct Stimulus
    {
        std::vector<float> GS;
        std::vector<float> GSD;
        std::string Type;
        std::string Variant;
        int Correct;
    };
    
    int LoadStimuli(std::string path, std::string format);
    void LoadStimulus(std::string path, std::string file, std::vector<Stimulus> &Stimulus);
}

#endif /* stimuli_hpp */
