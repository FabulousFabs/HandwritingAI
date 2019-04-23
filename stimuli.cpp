//
//  stimuli.cpp
//  CppCNNHandwriting
//
//  Created by Fabian Schneider on 23.04.19.
//  Copyright © 2019 Fabian Schneider. All rights reserved.
//

#include "stimuli.hpp"

using namespace stimuli;

int stimuli::LoadStimuli(std::string path, std::string format) {
    std::vector<std::string> Files = filesys::ScanDirectoryByFiletype(path, format);
    
    for (auto&& fn: Files) {
        std::cout << "Loading " << fn << "..." << std::endl;
        
        LoadStimulus(path, fn);
    }
    
    return 0;
}

int stimuli::LoadStimulus(std::string path, std::string file) {
    const std::string fn = path + file;
    cimg_library::CImg<unsigned char> image(fn.c_str());
    
    return 0;
}
