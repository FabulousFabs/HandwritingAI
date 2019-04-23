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

#include "filesys.hpp"

#define cimg_display 0
#define cimg_use_png
#include "CImg.h"

namespace stimuli {
    int LoadStimuli(std::string path, std::string format);
    int LoadStimulus(std::string path, std::string file);
}

#endif /* stimuli_hpp */