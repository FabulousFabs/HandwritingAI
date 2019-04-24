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
    std::vector<Stimulus> Stimuli;
    
    for (auto&& fn: Files) {
        std::cout << "Loading " << fn << "..." << std::endl;
        
        LoadStimulus(path, fn, Stimuli);
    }
    
    return 0;
}

void stimuli::LoadStimulus(std::string path, std::string file, std::vector<Stimulus> &Stimuli) {
    Stimulus stim;
    
    const std::string fp = path + file;
    cimg_library::CImg<unsigned char> image(fp.c_str());
    
    for (int y = 0; y < image.height(); y++) {
        for (int x = 0; x < image.width(); x++) {
            unsigned char *r = image.data(x, y, 0, 0),
                          *g = image.data(x, y, 0, 1),
                          *b = image.data(x, y, 0, 2);
            
            int R = (int) *r,
                G = (int) *g,
                B = (int) *b;
            
            float m = 255,
                  greyscale = 0.2126 * ((float) R / m) + 0.7152 * ((float) G / m) + 0.0722 * ((float) B / m),
                  greyscaledark = abs(1-greyscale);
            
            stim.GS.push_back(greyscale);
            stim.GSD.push_back(greyscaledark);
            
        }
    }
    
    Stimuli.push_back(stim);
}
