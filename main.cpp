//
//  main.cpp
//  CppCNNHandwriting
//
//  Created by Fabian Schneider on 23.04.19.
//  Copyright © 2019 Fabian Schneider. All rights reserved.
//

#include "main.h"

const std::string DirImages = "/users/fabianschneider/desktop/CppCNNHandwriting/CppCNNHandwriting/numbers/";
const std::string ImageFormat = "png";

int main (int argc, const char *argv[]) {
    stimuli::LoadStimuli(DirImages, ImageFormat);
    
    system("clear"); // libpng prints ugly iccp warnings bc it's stupid...
    std::cout << "Done." << std::endl;
    
    return 0;
}
