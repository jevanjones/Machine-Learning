// -----------------------------
// Backpropagation Class Functions
// -----------------------------

#include "backprop.h"
#include "error.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>

using std::vector;
using std::cout;

void Backprop::printAllNeurons(){
  for (int i = 0;i<neurons.size();i++){
    for (int j = 0;j<neurons[i].size();j++){
      neurons[i][j].toString();
    }
  }
}

// This function trains the perceptron with the given dataset
  void Backprop::train (Matrix& features, Matrix& labels){
      size_t numFeatures = features.cols();

      numOfHiddenNodes = 32;//numFeatures*2;

      features.shuffleRows(random,&labels); // and I shuffle my features so that I don't simply memorize the dataset, but actually learn

      Matrix trainingSet;
      Matrix trainingLabels;
      Matrix validationSet;
      Matrix validationLabels;

      trainingSet.copyPart(features,0,0,features.rows()*0.9,features.cols());
      validationSet.copyPart(features,features.rows()*0.9,0,features.rows()-features.rows()*0.9,features.cols());
      trainingLabels.copyPart(labels,0,0,labels.rows()*0.9,labels.cols());
      validationLabels.copyPart(labels,labels.rows()*0.9,0,labels.rows()-labels.rows()*0.9,labels.cols());

      //trainingSet.copyPart(features,0,0,features.rows(),2);

      for (int j = 0;j<numOfHiddenLayers;j++){ // initialize the layers of hidden nodes
        vector<Neuron> hidden;
        for (size_t i = 0;i<numOfHiddenNodes;i++){ // initialize the hidden nodes in the layer
          Neuron neu;
          neu.initWeights(numFeatures,random,1); // initialize the weights
          hidden.push_back(neu); // and add it to the list of the hidden layer
        }
        neurons.push_back(hidden); // add the newly created hidden layer to the list of nodes
      }

      vector<Neuron> outputs;

      for (size_t i = 0;i<labels.valueCount(0);i++){
        Neuron neu;
        neu.initWeights(numOfHiddenNodes,random,0); // initialize the weights to the number of hidden nodes that we have
        outputs.push_back(neu); // add the new neuron to the list of output neurons
      }
      neurons.push_back(outputs); // remember that the last value in the neurons vectpr are the outputs

      bestSoFar = neurons;

// The following is for the debug dataset. It isn't used for the other datsets
//      vector<double> dummy;
//
//      dummy.push_back(0.2);
//      dummy.push_back(-0.1);
//      dummy.push_back(0.1);
//
//      neurons[0][0].weights = dummy;
//
//      dummy.clear();
//
//      dummy.push_back(0.3);
//      dummy.push_back(-0.3);
//      dummy.push_back(-0.2);
//
//      neurons[0][1].weights = dummy;
//
//      dummy.clear();
//
//      dummy.push_back(-0.2);
//      dummy.push_back(-0.3);
//      dummy.push_back(0.1);
//
//      neurons[1][0].weights = dummy;
//
//      dummy.clear();
//
//      dummy.push_back(-0.1);
//      dummy.push_back(0.3);
//      dummy.push_back(0.2);
//
//      neurons[1][1].weights = dummy;
//
//      dummy.clear();
//
//      dummy.push_back(-0.1);
//      dummy.push_back(0.3);
//      dummy.push_back(0.2);
//
//      neurons[2][0].weights = dummy;
//
//      dummy.clear();
//
//      dummy.push_back(-0.2);
//      dummy.push_back(-0.3);
//      dummy.push_back(0.1);
//
//      neurons[2][1].weights = dummy;

      totalEpochs = 0; // initialize my number of epochs trained to 0
      double error = 1; // set the error to a vlue greater than 0.001, so that I can start training
      double prevError = 0;
      double validError = 1;
      double bestError = 1;
      double doomsDay = 0;
      while (doomsDay < 20){
        trainOneEpoch(trainingSet,trainingLabels); // I get the error back from training one epoch
        prevError = error;
        error = calculateError(trainingSet,trainingLabels);
        validError = calculateError(validationSet,validationLabels);
        cout << error << ",";
        cout << validError << ",";
        cout << validAccuracy << ",";

        //trainOneEpoch(trainingSet,labels);

        if (validError < bestError){
          bestSoFar = neurons;
          bestError = validError;
          doomsDay = 0;
        }
        else{
          doomsDay++;
        }

//        if (abs(error-prevError) < 0.001){
//          doomsDay++;
//        }
//        else{
//          doomsDay = 0;
//        }

        totalEpochs++; // I increment the number of epochs after I train once
        printf("%d\n",totalEpochs); // I print out the epoch iteration

        trainingSet.shuffleRows(random,&trainingLabels); // and I shuffle my features so that I don't simply memorize the dataset, but actually learn
      }
  }

// This function iterates through the training matrix once and updates weights accordingly
void Backprop::trainOneEpoch(Matrix& features, Matrix& labels){
    vector<vector<double> > outputs;
    vector<double> sigmas;
    vector<double> target;

//    vector<double> dummy;
//
//    dummy.push_back(0.1);
//    dummy.push_back(1);
//
    for (size_t k = 0;k<features.rows();k++){
      target.clear();

      for (int l = 0;l<labels.valueCount(0);l++){ // turn my target values into something I can use

        if (l == labels[k][0]){
          target.push_back(1);
        }
        else{
          target.push_back(0);
        }
      }

//      printAllNeurons();

//      cout << "Target Values: ";
//      for (int i = 0;i<dummy.size();i++){
//        cout << dummy[i] << ",";
//      }
//      cout << "\nInput: ";
//      for (int i = 0;i<features[k].size();i++){
//        cout << features[k][i] << ",";
//      }
//      cout << "\n";

      outputs.clear();
      vector<double> layerOut;
      for (int i = 0;i<neurons.size();i++){
        layerOut.clear();
        for (int j = 0;j<neurons[i].size();j++){
          if (i == 0){ // if I'm first inserting my input into the neural net, then I use the features values
            layerOut.push_back(neurons[i][j].getOutput(features[k]));
          }
          else{ // else we use the output of the previous layers as the input to the next layer
            layerOut.push_back(neurons[i][j].getOutput(outputs[i-1]));
          }
        }
        outputs.push_back(layerOut); // I save the output of my layer in the outputs vector
      }

//      cout << "Outputs: ";
//      for (int i = 0;i<outputs.size();i++){
//        for (int j = 0;j<outputs[i].size();j++){
//           cout<<outputs[i][j]<<",";
//        }
//        cout << "\n";
//      }
//      cout <<"\n";


      sigmas.clear();
      // Time to update the sigmas, after we've already calculated the outputs
      for (int i = neurons.size()-1;i>=0;i--){
        vector<double> tempSig;
        vector<double> tempWeights;
        for (int j = 0;j<neurons[i].size();j++){
          if (i == neurons.size()-1){ // If we're on the output layer then weights, etc, don't matter. only the target value
            neurons[i][j].calculateSigma(tempSig,tempWeights,target[j]); // sigmas and weights don't really matter here for this case
          }
          else{ // If we're on a hidden layer, then previous weights and sigmas matter, but the target doesn't
            for (int l = 0;l<neurons[i+1].size();l++){ // I collect the corresponding weights from the previous layer
              tempWeights.push_back(neurons[i+1][l].weights[j]);
              //cout << neurons[i+1][l].weights[j] << ",";
            }

            //cout << "Weights collected\n\n";
            neurons[i][j].calculateSigma(sigmas,tempWeights,0);
          }
          tempSig.push_back(neurons[i][j].sigma); // we save all of the sigma values for this particular layer
          tempWeights.clear(); // clear out the gathered weights for the next node in the list
        }
        sigmas.clear();
        sigmas = tempSig; // And update our sigmas for calculation in the next layer
//        cout << "What are my sigmas?";
//        for (int l = 0;l<sigmas.size();l++){
//          cout <<sigmas[l]<<",";
//        }
        tempSig.clear();
      }

      // Now that we've handled the sigmas, it's time to handle the weights
      for (int i = 0;i<neurons.size();i++){
        for (int j = 0;j<neurons[i].size();j++){
          if (i == 0){
            neurons[i][j].updateWeights(features[k],learningRate,momentum);
          }
          else{
            neurons[i][j].updateWeights(outputs[i-1],learningRate,momentum);
          }
        }
      }
    }
}

double Backprop::calculateError(Matrix& features, Matrix& labels){
    vector<vector<double> > outputs;
    vector<double> target;
    double error = 0;
    validAccuracy = 0;

    for (size_t k = 0;k<features.rows();k++){
      target.clear();
      for (int l = 0;l<labels.valueCount(0);l++){ // turn my target values into something I can use
        if (l == labels[k][0]){
          target.push_back(1);
        }
        else{
          target.push_back(0);
        }
      }

      outputs.clear();
      vector<double> layerOut;
      for (int i = 0;i<neurons.size();i++){
        layerOut.clear();
        for (int j = 0;j<neurons[i].size();j++){
          if (i == 0){ // if I'm first inserting my input into the neural net, then I use the features values
            layerOut.push_back(neurons[i][j].getOutput(features[k]));
          }
          else{ // else we use the output of the previous layers as the input to the next layer
            layerOut.push_back(neurons[i][j].getOutput(outputs[i-1]));
          }
        }
        outputs.push_back(layerOut); // I save the output of my layer in the outputs vector
      }

//      cout << "outputs: ";
//      for (int i = 0;i<outputs.size();i++){
//        for (int j = 0;j<outputs[i].size();j++){
//           cout<<outputs[i][j]<<",";
//        }
//        cout << "\n";
//      }
//      cout <<"\n";

      for (int i = 0;i<outputs[outputs.size()-1].size();i++){
        error += pow(target[i]-outputs[outputs.size()-1][i],2);
      }

      int outputsSize = outputs.size();
      double highest = 0;
      double sizeIndex = 0;

      for (int i = 0;i<outputs[outputsSize-1].size();i++){
        if (outputs[outputsSize-1][i]>highest){
          highest = outputs[outputsSize-1][i];
          sizeIndex = i;
        }
      }

      if (labels[k][0] == sizeIndex){
        validAccuracy++;
      }
    }



    validAccuracy /= ((double)labels.rows());

    error /= (double)features.rows();

    return error;
}

// This function predicts the result based on the training that had been given to the perceptron
  void Backprop::predict (const std::vector<double>& features,std::vector<double>& labels){
      vector<vector<double> > outputs;

      outputs.clear();
      vector<double> layerOut;
      for (int i = 0;i<bestSoFar.size();i++){
        layerOut.clear();
        for (int j = 0;j<bestSoFar[i].size();j++){
          if (i == 0){ // if I'm first inserting my input into the neural net, then I use the features values
            layerOut.push_back(bestSoFar[i][j].getOutput(features));
          }
          else{ // else we use the output of the previous layers as the input to the next layer
            layerOut.push_back(bestSoFar[i][j].getOutput(outputs[i-1]));
          }
        }
        outputs.push_back(layerOut); // I save the output of my layer in the outputs vector
      }


//      cout << "outputs: ";
//      for (int i = 0;i<outputs.size();i++){
//        for (int j = 0;j<outputs[i].size();j++){
//           cout<<outputs[i][j]<<",";
//        }
//        cout << "\n";
//      }
//      cout <<"\n";

      int outputsSize = outputs.size();
      double highest = 0;
      double sizeIndex = 0;

      for (int i = 0;i<outputs[outputsSize-1].size();i++){
        if (outputs[outputsSize-1][i]>highest){
          highest = outputs[outputsSize-1][i];
          sizeIndex = i;
        }
      }

      labels[0] = sizeIndex;
}
