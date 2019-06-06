//
//  stimuli.cpp
//  CppCNNHandwriting
//
//  Created by Fabian Schneider on 23.04.19.
//  Copyright © 2019 Fabian Schneider. All rights reserved.
//

#include "stimuli.hpp"

using namespace stimuli;

enum Type stimuli::GetTypeFromString (std::string &str) {
    for (int i = 0; i < Last; i++) {
        Type t = static_cast<Type>(i);
        if (GetTypeAsString(t) == str) {
            return t;
        }
    }
    return static_cast<Type>(0);
}

std::string stimuli::GetTypeAsString (enum Type &num) {
    return TypesAsStrings[num];
}

void stimuli::LoadStimuli(std::string path, std::string format, std::vector<Stimulus> &Stimuli) {
    std::vector<std::string> Files = filesys::ScanDirectoryByFiletype(path, format);
    
    for (auto&& fn: Files) {
        LoadStimulus(path, fn, Stimuli);
    }
}

void stimuli::LoadStimulus(std::string path, std::string file, std::vector<Stimulus> &Stimuli) {
    Stimulus stim;
    
    std::string fd = file.substr(0, file.find_last_of("."));
    std::regex strings("[^a-zA-Z]+");
    std::regex numbers("[^0-9]+");
    
    stim.Type = std::regex_replace(fd, strings, "");
    stim.Variant = std::regex_replace(fd, numbers, "");
    stim.Correct = GetTypeFromString(stim.Type);
    
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
            
            double m = 255.0,
                  greyscale = 0.2126 * ((double) R / m) + 0.7152 * ((double) G / m) + 0.0722 * ((double) B / m),
                  greyscaledark = abs(1-greyscale);
            
            stim.GS.push_back(greyscale);
            stim.GSD.push_back(greyscaledark);
        }
    }
    
    Stimuli.push_back(stim);
}
