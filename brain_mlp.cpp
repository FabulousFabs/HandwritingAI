//
//  brain_MLP.cpp
//  CppMLPHandwriting
//
//  Created by Fabian Schneider on 30.04.19.
//  Copyright © 2019 Fabian Schneider. All rights reserved.
//

#include "brain_mlp.hpp"

//
//
// CLASS brain::layer::flatten_proto

std::tuple<enum brain::MLP_LAYER_TYPE, int, enum brain::ACTIVATION_FUNC> brain::layer::Flatten(int n_args, ...) {
    va_list ap;
    va_start(ap, n_args);
    
    int neurons = 1;
    
    for (int i = 1; i < n_args; i++) {
        int n = va_arg(ap, int);
        std::cout << n << std::endl;
        neurons = neurons * n;
    }
    
    std::cout << neurons << std::endl;
    
    return std::tuple<enum brain::MLP_LAYER_TYPE, int, enum brain::ACTIVATION_FUNC>(MLP_LAYER_T_FLATTEN, neurons, ACTIVATION_SIGMOID);
}

brain::layer::flatten_proto::flatten_proto(int n, enum brain::ACTIVATION_FUNC af) {
    this->i_nNeurons = n;
    this->e_iActivationFunc = af;
}

void brain::layer::flatten_proto::Excite(brain::layer_proto *next_layer) {
    std::vector<std::vector<double>> neuronsOut;
    neuronsOut.push_back(this->v_fNeurons);
    this->v_fNeuronsOut = neuronsOut[0];
    
    for (int i = 0; i < this->v_fWeights[0].size() - 1; i++) {
        neuronsOut.push_back(this->v_fNeuronsOut);
    }
    neuronsOut = neuronsOut; // i x j
    std::vector<std::vector<double>> weights = this->v_fWeights; // j x i
    
    std::vector<std::vector<double>> p = brain::MatrixDot(neuronsOut, weights); // i x i (i times the output)
    
    next_layer->v_fNeurons = p[0];
}

//
//
// CLASS brain::layer::dense_proto

std::tuple<enum brain::MLP_LAYER_TYPE, int, enum brain::ACTIVATION_FUNC> brain::layer::Dense(int neurons, enum brain::ACTIVATION_FUNC af) {
    return std::tuple<enum brain::MLP_LAYER_TYPE, int, enum brain::ACTIVATION_FUNC>(MLP_LAYER_T_FLATTEN, neurons, af);
}

brain::layer::dense_proto::dense_proto(int n, enum brain::ACTIVATION_FUNC af) {
    this->i_nNeurons = n;
    this->e_iActivationFunc = af;
}

//
//
// CLASS brain::layer_proto

void brain::layer_proto::Grow() {
    for (int i = 0; i < this->i_nNeurons; i++) {
        this->v_fNeurons.push_back((double) 0);
        this->v_fNeuronsOut.push_back((double) 0);
    }
}

void brain::layer_proto::Neuroplasticity(enum brain::WEIGHTS_INIT wi, int ins, int outs) {
    for (int i = 0; i < this->v_fNeurons.size(); i++) {
        std::vector<double> p;
        for (int j = 0; j < outs; j++) {
            double p1 = (wi == WEIGHTS_INIT_RANDOM) ? brain::MakeRandomNP()
                        : (wi == WEIGHTS_INIT_XAVIER) ? brain::MakeRandomXavier(ins, outs)
                        : (brain::MakeRandomNP() / 10);
            p.push_back(p1);
        }
        this->v_fWeights.push_back(p);
    }
}

void brain::layer_proto::GetSensations(std::vector<double> s) {
    assert(s.size() == this->v_fNeurons.size());
    
    for (int i = 0; i < s.size(); i++) {
        this->v_fNeurons[i] = s[i];
    }
}

void brain::layer_proto::Excite(brain::layer_proto *next_layer) {
    std::vector<std::vector<double>> neurons;
    neurons.push_back(this->v_fNeurons);
    
    std::vector<std::vector<double>> neuronsOut = brain::Activate(neurons, this->e_iActivationFunc);
    this->v_fNeuronsOut = neuronsOut[0];
    
    for (int i = 0; i < this->v_fWeights[0].size() - 1; i++) {
        neuronsOut.push_back(this->v_fNeuronsOut);
    }
    neuronsOut = neuronsOut; // i x j
    std::vector<std::vector<double>> weights = this->v_fWeights; // j x i
    
    std::vector<std::vector<double>> p = brain::MatrixDot(neuronsOut, weights); // i x i (i times the output)
    
    next_layer->v_fNeurons = p[0];
}

void brain::layer_proto::Activate() {
    for (int i = 0; i < this->v_fNeurons.size(); i++) {
        this->v_fNeuronsOut[i] = brain::Activate(this->v_fNeurons[i], this->e_iActivationFunc);
    }
}

//
//
// CLASS brain::MLP

brain::MLP::MLP() {
    this->b_IsCompiled = false;
}

void brain::MLP::Sequential(std::tuple<enum brain::MLP_LAYER_TYPE, int, enum brain::ACTIVATION_FUNC> l) {
    assert(this->b_IsCompiled == false);
    
    if (std::get<0>(l) == MLP_LAYER_T_FLATTEN) {
        brain::layer::flatten_proto ls(std::get<1>(l), std::get<2>(l));
        this->Layers.push_back(ls);
    } else {
        brain::layer::dense_proto ls(std::get<1>(l), std::get<2>(l));
        this->Layers.push_back(ls);
    }
}

void brain::MLP::Compile(enum brain::WEIGHTS_INIT wi = brain::WEIGHTS_INIT_XAVIER, enum brain::optimiser::OPTIMISER_TYPE ot = brain::optimiser::OPTIMISER_ADAM, enum brain::optimiser::LOSS_FUNC lf = brain::optimiser::LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, double lr = 0.01) {
    assert(this->b_IsCompiled == false);
    
    // grow neurons
    for (int i = 0; i < this->Layers.size(); i++) {
        this->Layers[i].Grow();
    }
    
    // neuroplasticity
    for (int i = 0; i < this->Layers.size() - 1; i++) {
        int ins = (i > 0) ? (int) this->Layers[i - 1].v_fNeurons.size() : 0, outs = (int) this->Layers[i + 1].v_fNeurons.size();
        this->Layers[i].Neuroplasticity(wi, ins, outs);
    }
    
    this->e_iOptimiser = ot;
    this->e_iLossFunc = lf;
    this->f_LearningRate = lr;
    this->b_IsCompiled = true;
}

void brain::MLP::Flush() {
    for (int i = 0; i < this->Layers.size(); i++) {
        for (int n = 0; n < this->Layers[i].v_fNeurons.size(); n++) {
            this->Layers[i].v_fNeurons[n] = (double) 0;
            this->Layers[i].v_fNeuronsOut[n] = (double) 0;
        }
    }
}

std::tuple<int, double> brain::MLP::Perceive(std::vector<double> &s) {
    assert(this->b_IsCompiled == true);
    this->Flush();
    
    this->Layers[0].GetSensations(s);
    
    for (int i = 0; i < this->Layers.size() - 1; i++) {
        this->Layers[i].Excite(&this->Layers[i + 1]);
    }
    
    this->Layers[this->Layers.size() - 1].Activate();
    
    return this->GetChoice();
}

std::tuple<int, double> brain::MLP::GetChoice() {
    assert(this->b_IsCompiled == true);
    
    double max = -2;
    int maxN = 0;
    
    for (int i = 0; i < this->Layers[this->Layers.size() - 1].v_fNeuronsOut.size(); i++) {
        if (this->Layers[this->Layers.size() - 1].v_fNeuronsOut[i] > max) {
            max = this->Layers[this->Layers.size() - 1].v_fNeuronsOut[i];
            maxN = i;
        }
    }
    
    return std::tuple<int, double>(maxN, max);
}

void brain::MLP::Feedback(int correct) {
    int percept = std::get<0>(this->GetChoice());
    
    if (this->e_iOptimiser == brain::optimiser::OPTIMISER_SGD) {
        this->StochasticGradientDescentOptimisation(percept, correct);
    }
}

void brain::MLP::StochasticGradientDescentOptimisation(int percept, int correct) {
    // initialize
    std::vector<std::vector<double>> loss;
    for (int i = 0; i < this->Layers.size(); i++) {
        loss.push_back(std::vector<double>());
    }
    
    //std::cout << "p=" << percept << "; c=" << correct << std::endl;
    
    // get output loss
    loss[this->Layers.size() - 1] = brain::Loss(this->Layers[this->Layers.size() - 1].v_fNeuronsOut, correct, this->e_iLossFunc);
    
    //xy::print(loss);
    
    // chain rule loss + theta correction
    for (int i = (int) this->Layers.size() - 2; i >= 0; i--) {
        for (int n = 0; n < this->Layers[i].v_fNeurons.size(); n++) {
            double e = 0;
            for (int j = 0; j < this->Layers[i+1].v_fNeurons.size(); j++) {
                //e += (this->Layers[i].v_fWeights[n][j] * loss[i+1][j]) * brain::Derive(this->Layers[i].v_fNeuronsOut[n], this->Layers[i].e_iActivationFunc);
                e = e + (this->Layers[i].v_fWeights[n][j] * loss[i+1][j]);
            }
            //loss[i].push_back((e / (double) this->Layers[i+1].v_fNeurons.size()));
            loss[i].push_back(e);
            
            // theta
            for (int j = 0; j < this->Layers[i+1].v_fNeurons.size(); j++) {
                double theta = brain::DeriveLoss(loss[i][n], this->e_iLossFunc);
                this->Layers[i].v_fWeights[n][j] = this->Layers[i].v_fWeights[n][j] - (theta * this->f_LearningRate);
                
                /*if (i == 1 && n == 50 && j == 3) {
                    std::cout << "W=" << this->Layers[i].v_fWeights[n][j];
                    std::cout << " (T=" << theta << ")";
                    std::cout << " -> (Wn=" << (this->Layers[i].v_fWeights[n][j] - (theta * this->f_LearningRate)) << ")";
                    std::cout << std::endl;
                }*/
            }
        }
    }
    
    
    double lt = 0.0;
    for (int i = 0; i < loss[this->Layers.size() - 1].size(); i++) {
        lt = lt + loss[this->Layers.size() - 1][i];
    }
    lt = lt / (double) loss[this->Layers.size() - 1].size();
    std::cout << "Loss=" << lt << std::endl;
    
    //xy::print(loss);

    /*
    // recalculate theta
    for (int i = 0; i < this->Layers.size(); i++) {
        for (int n = 0; n < this->Layers[i].v_fNeurons.size(); n++) {
            for (int j = 0; )
        }
    }*/
}

/*
void brain::MLP::StochasticGradientDescentOptimisation(int percept, int correct) {
    std::vector<double> errors_o = brain::Loss(this->Layers[this->Layers.size() - 1].v_fNeuronsOut, correct, this->e_iLossFunc);
    
    std::vector<std::vector<double>> error;
    std::vector<std::vector<double>> delta;
    for (int i = 0; i < this->Layers.size(); i++) {
        std::vector<double> v;
        std::vector<double> v2;
        for (int j = 0; j < this->Layers[i].v_fNeuronsOut.size(); j++) {
            if (i == this->Layers.size() - 1) {
                v.push_back(errors_o[j] * brain::Derive(this->Layers[i].v_fNeurons[j], this->Layers[i].e_iActivationFunc));
                v2.push_back(errors_o[j]);
            } else {
                v.push_back((double) 0);
                v2.push_back((double) 0);
            }
        }
        error.push_back(v2);
        delta.push_back(v);
    }
    
    for (int i = (int) this->Layers.size() - 2; i > 0; i--) {
        for (int j = 0; j < this->Layers[i].v_fNeuronsOut.size(); j++) {
            double error_n = 0, delta_n = 0;
            
            for (int k = 0; k < this->Layers[i].v_fWeights[j].size(); k++) {
                error_n += error[i + 1][k] * this->Layers[i].v_fWeights[j][k] * brain::Derive(this->Layers[i].v_fNeurons[j], this->Layers[i].e_iActivationFunc);
                //std::cout << error_n << std::endl;
            }
            
            delta_n = error_n * brain::Derive(this->Layers[i].v_fNeuronsOut[j], this->Layers[i].e_iActivationFunc);
            error[i][j] = error_n;
            delta[i][j] = delta_n;
        }
    }
    
    for (int i = 0; i < this->Layers.size() - 1; i++) {
        for (int j = 0; j < this->Layers[i].v_fNeuronsOut.size(); j++) {
            for (int k = 0; k < this->Layers[i].v_fWeights[j].size(); k++) {
                this->Layers[i].v_fWeights[j][k] = this->Layers[i].v_fWeights[j][k] - (this->f_LearningRate * delta[i][j]);
            }
        }
    }
}
*/

std::tuple<int, double> brain::MLP::Train(std::vector<double> &s, int correct) {
    std::tuple<int, double> p = this->Perceive(s);
    this->Feedback(correct);
    return p;
}
