//
//  main.cpp
//  CppMLPHandwriting
//
//  Created by Fabian Schneider on 23.04.19.
//  Copyright © 2019 Fabian Schneider. All rights reserved.
//

#include "main.h"

const std::string DirImages = "/users/fabianschneider/desktop/CppCNNHandwriting/CppCNNHandwriting/numbers/";
const std::string ImageFormat = "png";

namespace xy {
    void print(std::vector<std::vector<double>> m) {
        for (int i = 0; i < m.size(); i++) {
            for (int j = 0; j < m[i].size(); j++) {
                std::cout << m[i][j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

int main (int argc, const char *argv[]) {
    std::vector<stimuli::Stimulus> Stimuli;
    stimuli::LoadStimuli(DirImages, ImageFormat, Stimuli);
    
    system("clear"); // libpng prints ugly iccp warnings bc it's stupid...
    std::cout << "Stimuli are now loaded." << std::endl;
    
    /*std::cout << "Stimlen=" << Stimuli[0].GSD.size() << std::endl;
    
    
    return 0;*/
    
    /*brain::MLP MLP;
    MLP.Sequential(brain::layer::Flatten(3, 16, 16, 1));
    //MLP.Sequential(brain::layer::Dense(64, brain::ACTIVATION_RELU));
    MLP.Sequential(brain::layer::Dense(64, brain::ACTIVATION_TANH));
    //MLP.Sequential(brain::layer::Flatten(1, 10));
    MLP.Sequential(brain::layer::Dense(10, brain::ACTIVATION_SIGMOID));
    MLP.Compile(
                brain::WEIGHTS_INIT_XAVIER,
                brain::optimiser::OPTIMISER_SGD,
                brain::optimiser::LOSS_SPARSE_CATEGORICAL_CROSSENTROPY,
                0.001
    );*/
    
    /*brain::MLP MLP;
    MLP.Sequential(brain::layer::Flatten(3, 16, 16, 1));
    MLP.Sequential(brain::layer::Dense(64, brain::ACTIVATION_RELU));
    MLP.Sequential(brain::layer::Dense(64, brain::ACTIVATION_TANH));
    MLP.Sequential(brain::layer::Dense(32, brain::ACTIVATION_RELU));
    MLP.Sequential(brain::layer::Dense(32, brain::ACTIVATION_TANH));
    MLP.Sequential(brain::layer::Dense(16, brain::ACTIVATION_RELU));
    MLP.Sequential(brain::layer::Dense(16, brain::ACTIVATION_TANH));
    MLP.Sequential(brain::layer::Dense(10, brain::ACTIVATION_SIGMOID));
    MLP.Compile(
                brain::WEIGHTS_INIT_XAVIER,
                brain::optimiser::OPTIMISER_SGD,
                brain::optimiser::LOSS_SQUARE,
                0.01
                );*/
    
    brain::MLP MLP;
    MLP.Sequential(brain::layer::Flatten(3, 16, 16, 1));
    MLP.Sequential(brain::layer::Dense(64, brain::ACTIVATION_TANH));
    MLP.Sequential(brain::layer::Dense(10, brain::ACTIVATION_TANH));
    MLP.Compile(
                brain::WEIGHTS_INIT_XAVIER,
                brain::optimiser::OPTIMISER_SGD,
                brain::optimiser::LOSS_SQUARE,
                0.01
    );
    
    int epoch = 0;
    double alpha = 0.95;
    double alpha_c = 0.00;
    
    while (alpha_c < alpha) {
        epoch++;
        
        int success = 0;
        for (auto&& s:Stimuli) {
            std::tuple<int, double> p = MLP.Train(s.GSD, (int) s.Correct);
            if (std::get<0>(p) == (int) s.Correct) {
                success++;
            }
        }
        
        alpha_c = (double) success / (double) Stimuli.size();
        
        std::cout << "Epoch" << epoch << " done. Current alpha=" << alpha_c << " (success=" << success << ")." << std::endl;
    }
    
    return 0;
}
