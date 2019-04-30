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
    cnn.Sequential(brain::layer::Dense(64, brain::ACTIVATION_TANH));
    cnn.Sequential(brain::layer::Dense(10, brain::ACTIVATION_SOFTPLUS));
    cnn.Compile();
    
    cnn.Perceive(Stimuli[0].GS);
    
    /*
     // some brain::matrix tests
     
    std::vector<std::vector<float>> m1;
    std::vector<float> m1_1;
    m1_1.push_back(1);
    m1_1.push_back(2);
    std::vector<float> m1_2;
    m1_2.push_back(3);
    m1_2.push_back(4);
    m1.push_back(m1_1);
    m1.push_back(m1_2);
    
    std::vector<std::vector<float>> m2;
    std::vector<float> m2_1;
    m2_1.push_back(5);
    m2_1.push_back(6);
    std::vector<float> m2_2;
    m2_2.push_back(7);
    m2_2.push_back(8);
    m2.push_back(m2_1);
    m2.push_back(m2_2);
    
    std::cout << "Before:" << std::endl;
    std::cout << "M1:" << std::endl;
    print(m1);
    std::cout << "M2:" << std::endl;
    print(m2);
    
    brain::MatrixFit(m1, m2);
    
    std::cout << "After:" << std::endl;
    std::cout << "M1:" << std::endl;
    print(m1);
    std::cout << "M2:" << std::endl;
    print(m2);
    
    std::cout << "M2.T:" << std::endl;
    std::vector<std::vector<float>> m3 = brain::MatrixT(m2);
    print(m3);
    
    std::vector<std::vector<float>> m4 = brain::MatrixDot(m1, m2);
    std::cout << "Product:" << std::endl;
    print(m4);*/
    
    //std::vector<int> circuit_structure = brain::MakeCircuitVector(3, 256, 64, 10);
    //brain::CNN cnn(circuit_structure, 0.5, brain::ACTIVATION_TANH);
    
    return 0;
}
