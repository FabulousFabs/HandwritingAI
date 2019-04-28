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
    std::vector<stimuli::Stimulus> Stimuli;
    stimuli::LoadStimuli(DirImages, ImageFormat, Stimuli);
    
    system("clear"); // libpng prints ugly iccp warnings bc it's stupid...
    std::cout << "Stimuli are now loaded." << std::endl;
    
    std::vector<int> circuit_structure = brain::MakeCircuitVector(3, 256, 64, 10);
    brain::CNN cnn(circuit_structure, 0.5, brain::ACTIVATION_TANH);
    
    
    return 0;
}
