// -------------------------------------
// Perceptron Learner Class
// Created by: John Evan Jones
// Class: CS 478 Section 1
// Date of First Creation: 1/24/18
// -------------------------------------

#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <iostream>
#include <stdio.h>
#include <vector>
#include "matrix.h"
#include "learner.h"
#include "rand.h"
#include "error.h"

using namespace std;

class Perceptron : public SupervisedLearner{
private:
  Rand& random;
  class Neuron{
  public:
    vector<double> weights;
    int neuronSize;

    Neuron(){}
    ~Neuron(){}

    double calculateNet(const vector<double>& nodes){
      double net = 0;
        for (int i = 0; i <weights.size();i++){
          if (i == (weights.size() - 1)){ // if we've reached the last value in the wieghts vector, we've reached the bias
            net += (weights[i]*1);
          }
          else{
            net += (weights[i]*nodes[i]);
          }
        }
        return net;
    }

    double checkTrigger(const vector<double>& nodes){
      if (calculateNet(nodes) > 0){
        return 1;
      } else {
        return 0;
      }
    }
  };

  Neuron * neuron;
  int totalEpochs;
  double learningRate;

  double trainOneEpoch(Matrix& features, Matrix&labels);
public:
  Perceptron (Rand& r) : SupervisedLearner(), random(r){
    neuron = new Neuron();
    learningRate = 0.1;
  }
  ~Perceptron() {
    if (neuron != NULL)
      delete neuron;
  }

  void train (Matrix& features, Matrix& labels);

  void predict (const std::vector<double>& features,std::vector<double>& labels);
};

#endif
