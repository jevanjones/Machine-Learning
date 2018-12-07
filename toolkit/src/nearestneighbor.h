
// -------------------------------------
// Perceptron Learner Class
// Created by: John Evan Jones
// Class: CS 478 Section 1
// Date of First Creation: 1/24/18
// -------------------------------------

#ifndef NEARESTNEIGHBOR_H
#define NEARESTNEIGHBOR_H

#include <iostream>
#include <stdio.h>
#include <vector>
#include "matrix.h"
#include "learner.h"
#include "rand.h"
#include "error.h"

using namespace std;

struct Neighbor{
  double distance;
  double output;
};

class NearestNeighbor : public SupervisedLearner{
private:
  Rand& random;
  vector<Neighbor> neighbors;
  int valueK;
  Matrix dataset;
  Matrix dataLabels;
  vector<size_t> attrValues;
  size_t labelValue;
  vector<double> mins;
  vector<double> maxes;

  Neighbor calculateDistance(const vector<double>& input,vector<double>& testCase,double& tLabel);
  void normalizeData();
  vector<double> normalizeVector(const vector<double>& input);
  double regressionOutput();
  double regressionOutputWeighted();
  double classificationOutput();
  double classificationOutputWeighted();
public:
  NearestNeighbor (Rand& r) : SupervisedLearner(), random(r){
    valueK = 9;
  }
  ~NearestNeighbor() {}

  void train (Matrix& features, Matrix& labels);

  void predict (const std::vector<double>& features,std::vector<double>& labels);
};

#endif
