
// -------------------------------------
// Clustering Learner Class
// Created by: John Evan Jones
// Class: CS 478 Section 1
// Date of First Creation: 3/27/18
// -------------------------------------

#ifndef CLUSTERING_H
#define CLUSTERING_H

#include <iostream>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <math.h>
#include <map>
#include "matrix.h"
#include "learner.h"
#include "rand.h"
#include "error.h"

using namespace std;


class Clustering : public SupervisedLearner{
private:
  Rand& random;
  int valK;
  double prevError;
  vector<size_t> attrValues;

  class Cluster{
  public:
    int clusterID;
    vector<double> centroid;
    vector< vector<double> > data;

    Cluster(){}
    Cluster(const Cluster& cluster){
      clusterID = cluster.clusterID;
      centroid = cluster.centroid;
      data = cluster.data;
    }
    ~Cluster(){}
  };

  vector<Cluster> clusters;

  double calculateError();
  double columnMean(int col, vector< vector<double> > info);
  double mostCommonValue(int col, vector< vector<double> > info);
  double calculateDistance(vector<double>& first,vector<double>& second);
  double silhouette(vector<double>& ins,int cID);
  double aScore(vector<double>& ins,int cID);
  double bScore(vector<double>& ins,int cID);
public:
  Clustering(Rand& r) : SupervisedLearner(), random(r){
    valK = 7;
    prevError = 0;
  }
  ~Clustering() {}

  void train (Matrix& features, Matrix& labels);

  void predict (const std::vector<double>& features,std::vector<double>& labels);
};

#endif
