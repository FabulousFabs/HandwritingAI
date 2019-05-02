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

void print(std::vector<std::vector<float>> m) {
    for (int i = 0; i < m.size(); i++) {
        for (int j = 0; j < m[i].size(); j++) {
            std::cout << m[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main (int argc, const char *argv[]) {
    std::vector<stimuli::Stimulus> Stimuli;
    stimuli::LoadStimuli(DirImages, ImageFormat, Stimuli);
    
    system("clear"); // libpng prints ugly iccp warnings bc it's stupid...
    std::cout << "Stimuli are now loaded." << std::endl;
    
    brain::CNN cnn;
    cnn.Sequential(brain::layer::Flatten(3, 16, 16, 1));
    cnn.Sequential(brain::layer::Dense(64, brain::ACTIVATION_RELU_LEAKY));
    cnn.Sequential(brain::layer::Dense(10, brain::ACTIVATION_SOFTPLUS));
    cnn.Compile(
                brain::WEIGHTS_INIT_XAVIER,
                brain::optimiser::OPTIMISER_SGD,
                brain::optimiser::LOSS_SQUARE,
                0.5
                );
    
    int epoch = 0;
    float alpha = 0.95;
    float alpha_c = 0.00;
    
    while (alpha_c < alpha) {
        epoch++;
        
        int success = 0;
        for (auto&& s:Stimuli) {
            std::tuple<int, float> p = cnn.Train(s.GSD, (int) s.Correct);
            if (std::get<0>(p) == (int) s.Correct) {
                success++;
            }
        }
        
        alpha_c = (float) success / (float) Stimuli.size();
        
        std::cout << "Epoch" << epoch << " done. Current alpha=" << alpha_c << " (success=" << success << ")." << std::endl;
    }
    
    return 0;
}
