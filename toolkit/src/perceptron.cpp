// -----------------------------
// Perceptron Class Functions
// -----------------------------

#include "perceptron.h"
#include "error.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>

using std::vector;
using std::cout;


// This function trains the perceptron with the given dataset
  void Perceptron::train (Matrix& features, Matrix& labels){

      size_t numFeatures = features.cols()+1;

      for (size_t i = 0;i<numFeatures;i++){
          double dummy = 0;
          neuron->weights.push_back(dummy);
      }

      for (int i = 0; i<neuron->weights.size();i++){
      cout << neuron->weights[i] << " ";
      }

      totalEpochs = 0;
      while ((trainOneEpoch(features,labels) > 0.001) || totalEpochs < 5){
        totalEpochs++;
        printf("Total Epochs: %d\n",totalEpochs);

        features.shuffleRows(random,&labels);
      }

      printf("Final Weight Vector: ");
      for (int i = 0; i < neuron->weights.size();i++){
        cout << neuron->weights[i] << " ";
      }
  }

// This function iterates through the training matrix once and updates weights accordingly
double Perceptron::trainOneEpoch(Matrix& features, Matrix& labels){
    double trigger;
    double error = 0;

    for (size_t i = 0; i<features.cols();i++){
      trigger = neuron->checkTrigger(features[i]);

        double dW = 0;

        error += pow(abs(labels[i][0]-trigger),2); // take the SSE

        for (int j = 0; j <neuron->weights.size();j++){
          if (j == (neuron->weights.size() - 1)){
            dW = learningRate*(labels[i][0]-trigger)*1;
          }
          else{
            dW = learningRate*(labels[i][0]-trigger)*features[i][j];
          }
            neuron->weights[j] += dW;
        }
    }

    return error;
}

// This function predicts the result based on the training that had been given to the perceptron
  void Perceptron::predict (const std::vector<double>& features,std::vector<double>& labels){
      for (size_t i = 0;i<labels.size();i++){
        labels[i] = neuron->checkTrigger(features);
      }
  }
