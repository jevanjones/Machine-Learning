//------------------------------------
// Nearest Neighbor Class Functions
//------------------------------------
#include "nearestneighbor.h"
#include "error.h"
#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cmath>

using std::vector;
using std::cout;

#define UNKNOWN_VALUE -1e308
#define ZERO_EQUIVALENT 0.000000000001

void NearestNeighbor::train (Matrix& features, Matrix& labels){
  //cout << "In train...lol\n";
  dataset = features;
  dataLabels = labels;
  //cout << "Saving what type of values the attributes have\n";
  for (size_t i = 0;i<features.cols();i++){
    //cout << "Saving the value in col " << i << " with value count of " << features.valueCount(i) << "\n";
    attrValues.push_back(features.valueCount(i)); // save whether or not the input is continuous, binary, etc
  }
  //normalizeData();
  //cout << "Also saving the label value count\n";
  labelValue = labels.valueCount(0);
}

void NearestNeighbor::predict (const std::vector<double>& features,std::vector<double>& labels){
  //cout << "Now in predict\n";
  if (!neighbors.empty()){
    neighbors.clear();
  }

  for (size_t i = 0;i<dataset.rows();i++){
    if (neighbors.size()==valueK){
      Neighbor n;
      n = calculateDistance(features,dataset[i],dataLabels[i][0]);
      for (int j = 0;j<neighbors.size();j++){
          if (n.distance < neighbors[j].distance){
            neighbors[j] = n;
            break;
          }
      }
    }
    else{
      //cout << "Filling neighbor vector\n";
      neighbors.push_back(calculateDistance(features,dataset[i],dataLabels[i][0]));
    }
  }

//  for (int i = 0;i<neighbors.size();i++){
//    cout << "Neighbor: " << i << " Output: " << neighbors[i].output << " Distance: " << neighbors[i].distance << "\n";
//  }

  labels[0] = regressionOutputWeighted();
}

vector<double> NearestNeighbor::normalizeVector(const vector<double>& input){
    vector<double> normal;
    for (int i = 0;i<input.size();i++){
      normal.push_back((input[i]-mins[i])/(maxes[i]-mins[i]));
    }
    return normal;
}

void NearestNeighbor::normalizeData(){
  mins.clear();
  maxes.clear();
  for (size_t j = 0;j<dataset.cols();j++){
    mins.push_back(dataset.columnMin(j));
    maxes.push_back(dataset.columnMax(j));
  }

  for (size_t i = 0;i<dataset.rows();i++){
    for (size_t j = 0;j<dataset.cols();j++){
      dataset[i][j] = (dataset[i][j]-mins[j])/(maxes[j]-mins[j]);
    }
  }
}

Neighbor NearestNeighbor::calculateDistance(const vector<double>& input,vector<double>& testCase,double& tLabel){
  Neighbor n;
  double sum = 0;

  vector<double> in;
  in = input;//normalizeVector(input);

  //cout << "Calculating distance\n";

  for (int i = 0;i<in.size();i++){
    if(attrValues[i] == 0){
      if(in[i] == UNKNOWN_VALUE || testCase[i] == UNKNOWN_VALUE){
        sum++;
      }
      else{
        //sum += pow(in[i]-testCase[i],2); // euclidean distance
        if (in[i] != 0 && testCase[i] != 0){
          sum += abs(in[i]-testCase[i])/abs(in[i]+testCase[i]);
        }
      }
    }
    else{
      if (in[i] != testCase[i] || in[i] == UNKNOWN_VALUE || testCase[i] == UNKNOWN_VALUE){
        sum++;
      }
    }
  }

  n.output = tLabel;
  //n.distance = sqrt(sum); // euclidean Distance
  n.distance = sum;

  return n;
}

double NearestNeighbor::regressionOutput(){
  double sum = 0;
  for (int i = 0;i<neighbors.size();i++){
    sum += neighbors[i].output;
  }
  sum /= valueK;
  return sum;
}


double NearestNeighbor::regressionOutputWeighted(){
  double sum = 0;
  double baseline = 0;
  for (int i = 0;i<neighbors.size();i++){
    if (neighbors[i].distance == 0){
      neighbors[i].distance = ZERO_EQUIVALENT;
    }
    sum += (neighbors[i].output/pow(neighbors[i].distance,2));
    baseline += 1/pow(neighbors[i].distance,2);
  }

  sum /= baseline;
  return sum;
}

double NearestNeighbor::classificationOutput(){
  double winner = 0;
  double highest = 0;

  //cout << "Classifying output\n";

  for (double i = 0;i<(double)labelValue;i++){
    double sum = 0;
    for (int j = 0;j<neighbors.size();j++){
      if(neighbors[j].output == i){
        sum++;
      }
    }
    sum /= valueK;
    if (sum > highest){
      highest = sum;
      winner = i;
    }
  }
  return winner;
}

double NearestNeighbor::classificationOutputWeighted(){
  double winner = 0;
  double highest = 0;
  double baseline = 0;

  for (int i = 0;i<neighbors.size();i++){
    if (neighbors[i].distance == 0){
      neighbors[i].distance = ZERO_EQUIVALENT;
    }
    baseline += 1/pow(neighbors[i].distance,2); // calculate the denominator for the classification output equation
  }

  for (double i = 0;i<(double)labelValue;i++){
    double sum = 0;
    for (int j = 0;j<neighbors.size();j++){
      if(neighbors[j].output == i){
        sum += 1/pow(neighbors[j].distance,2);
      }
    }
    sum /= baseline;
    if (sum > highest){
      highest = sum;
      winner = i;
    }
  }
  return winner;
}
