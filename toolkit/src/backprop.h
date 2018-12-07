// -------------------------------------
// Backpropagation Learner Class
// Created by: John Evan Jones
// Class: CS 478 Section 1
// Date of First Creation: 2/8/18
// -------------------------------------

#ifndef BACKPROP_H
#define BACKPROP_H

#include <iostream>
#include <stdio.h>
#include <vector>
#include <math.h>
#include "matrix.h"
#include "learner.h"
#include "rand.h"
#include "error.h"

using namespace std;

class Backprop : public SupervisedLearner{
private:
  Rand& random;
  class Neuron{
  public:
    vector<double> weights;
    vector<double> prevDeltaW;
    double net;
    double output;
    double sigma;
    bool isHidden;

    Neuron(){}
    ~Neuron(){}

    void initWeights(size_t featuresNum, Rand& r,bool hidden){
      for (size_t i = 0;i<featuresNum+1;i++){
          weights.push_back(r.uniform());
          prevDeltaW.push_back(0);
      }
      isHidden = hidden;
    }

    double calculateNet(const vector<double>& nodes){
      net = 0;
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

    double getOutput(const vector<double>& nodes){
      calculateNet(nodes);
      output = 1/(1+exp(-net));
      return output;
    }

    void updateWeights(vector<double>& outputs,double learn,double moment){
      vector<double> dwTemp;
      double dW = 0;

      for (int i = 0;i<weights.size();i++){
        if (i == weights.size()-1){
          dW = learn*1*sigma+moment*prevDeltaW[i]; // Account for the bias
        }
        else{
          dW = learn*outputs[i]*sigma+moment*prevDeltaW[i];
        }

        weights[i] += dW;
        dwTemp.push_back(dW);
      }

      prevDeltaW.clear();
      prevDeltaW = dwTemp;
    }

    double calculateSigma(vector<double> sigmas,vector<double> layerWeights, double target){
      sigma = 0;
      if (isHidden){
        double sum = 0;
        for (int i = 0;i<layerWeights.size();i++){
          sum += (layerWeights[i]*sigmas[i]);
        }
        sigma = sum * (output*(1-output));
      }
      else{
        sigma = (target-output)*(output*(1-output));
      }
      return sigma;
    }

    void toString(){
      cout.precision(10);
      cout << "Neruon Details:\nisHidden: " << isHidden << "\n";
      cout << "Output: " << output << "\n";
      cout << "Sigma: " << sigma << "\n";

      cout << "Current Weights: " << "\n";
      for (int i = 0;i<weights.size();i++){
        cout << weights[i] << ",";
      }
      cout << "\n";

      cout << "Delta Weights: " << "\n";
      for (int i = 0;i<prevDeltaW.size();i++){
        cout << prevDeltaW[i] << ",";
      }
      cout << "\n\n";
    }
  };

  vector<vector<Neuron> > neurons;
  vector<vector<Neuron> > bestSoFar;
  int totalEpochs;
  double learningRate;
  double momentum;
  int numOfHiddenLayers;
  size_t numOfHiddenNodes;

  double validAccuracy;

  void trainOneEpoch(Matrix& features, Matrix& labels);
  double calculateError(Matrix& features, Matrix& labels);
  void printAllNeurons();
public:
  Backprop(Rand& r) : SupervisedLearner(), random(r){
    learningRate = 0.5;
    momentum = 0.5;
    numOfHiddenLayers = 1;
  }
  ~Backprop() {}

  void train (Matrix& features, Matrix& labels);

  void predict (const std::vector<double>& features,std::vector<double>& labels);
};

#endif
