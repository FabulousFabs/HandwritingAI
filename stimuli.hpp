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
#include <regex>

#include "filesys.hpp"

#define cimg_display 0
#define cimg_use_png
#include "CImg.h"

namespace stimuli {
    enum Type
    {
        Zero = 0, One, Two, Three, Four, Five, Six, Seven, Eight, Nine,
        Last = Nine
    };
    
    static const std::string TypesAsStrings[10] = {"ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"};
    
    struct Stimulus
    {
        std::vector<float> GS;
        std::vector<float> GSD;
        std::string Type;
        std::string Variant;
        enum Type Correct;
    };
    
    void LoadStimuli(std::string path, std::string format, std::vector<Stimulus> &Stimulus);
    void LoadStimulus(std::string path, std::string file, std::vector<Stimulus> &Stimulus);
    std::string GetTypeAsString(enum Type &num);
    enum Type GetTypeFromString(std::string &str);
}

#endif /* stimuli_hpp */
