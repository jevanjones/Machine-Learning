#include "clustering.h"

#include <iostream>
#include <iomanip>
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

using std::vector;
using std::cout;

#define UNKNOWN_VALUE -1e308
#define ZERO_EQUIVALENT 0.000000000001

void Clustering::predict(const std::vector<double>& features,std::vector<double>& labels){}

double Clustering::columnMean(int col,vector< vector<double> > info) //Yes, it was copied from the toolkit. I did this for ease of use
{
	double sum = 0.0;
	size_t count = 0;
	std::vector< std::vector<double> >::iterator it;
	for(it = info.begin(); it != info.end(); it++)
	{
		double val = (*it)[col];
		if(val != UNKNOWN_VALUE)
		{
			sum += val;
			count++;
		}
	}
	if (count == 0){
    cout  << "Entire real column unknowns\n";
		return UNKNOWN_VALUE; // if all of the values in the column are unknown, we simply return unknown
	}
	return sum / count;
}

double Clustering::mostCommonValue(int col, vector< vector<double> > info) //Yes, it was copied from the toolkit. I did this for ease of use
{
	map<double, size_t> counts;
	vector< vector<double> >::iterator it;
	for(it = info.begin(); it != info.end(); it++)
	{
		double val = (*it)[col];
		if(val != UNKNOWN_VALUE)
		{
			map<double, size_t>::iterator pair = counts.find(val);
			if(pair == counts.end())
				counts[val] = 1;
			else
				pair->second++;
		}
	}
	size_t valueCount = 0;
	double value = 0;
	for(map<double, size_t>::iterator i = counts.begin(); i != counts.end(); i++)
	{
		if(i->second > valueCount)
		{
			value = i->first;
			valueCount = i->second;
		}
	}
  if (valueCount == 0){
    cout << "Entire nominal column unknowns\n";
    return UNKNOWN_VALUE;
  }
	return value;
}

void Clustering::train(Matrix& features, Matrix& labels){
    for (int i = 0;i<valK;i++){ // initialize my clusters
      uint64 temp = random.next(features.rows());
      Cluster cluster;
      cluster.clusterID = i;
      cluster.centroid = features[temp]; // the initial centroid is the same as the initial point in the cluster
      cluster.data.push_back(features[temp]);
      clusters.push_back(cluster);
    }

    for (size_t i = 0;i<features.cols();i++){
      attrValues.push_back(features.valueCount(i));
    }

    double iterationCount = 1;
    double error = 1;

    cout << "Num of clusters: " << clusters.size() << "\n";

    while (prevError != error || (abs((int)error-(int)prevError))> 0.0001){
      prevError = error;

      cout << "************\n Iteration " << iterationCount << " \n************\n\n";

      for (int i = 0;i<clusters.size();i++){
        cout << "Cluster " << i << " : Centroid: ";
        for (int j = 0;j<clusters[i].centroid.size();j++){
          cout << clusters[i].centroid[j] << ", ";
        }
        cout << "\n with " << clusters[i].data.size() << " instances\n";
      }

      for (int i = 0;i<clusters.size();i++){ // empty out the clusters data for this iteration
          clusters[i].data.clear();
      }

      double newLineCounter = 0;
      cout << "     ";
      for (size_t i = 0;i<features.rows();i++){
        double lowest = 100000;
        double lowIndex = 0;
        for (int j = 0;j<clusters.size();j++){
          double temp = calculateDistance(features[i],clusters[j].centroid);
          if (temp < lowest){
            lowest = temp;
            lowIndex = j;
          }
        }
        cout << i << "=" << lowIndex << " ";
        newLineCounter++;
        if (newLineCounter == 10){
          newLineCounter = 0;
          cout << "\n   ";
        }
        clusters[lowIndex].data.push_back(features[i]);
      }

      cout << "\n";
      error = calculateError();

      for (int i = 0;i<clusters.size();i++){
        for (int j = 0;j<clusters[i].centroid.size();j++){
          if (attrValues[j] == 0){
            clusters[i].centroid[j] = columnMean(j,clusters[i].data);
          }
          else{
            clusters[i].centroid[j] = mostCommonValue(j,clusters[i].data);
          }
        }
      }
      cout << "SSE for this iteration: " << error << "\n\n";
      iterationCount++;
    }
    cout << "SSE has converged\n";

    double sil = 0;
    for (int i = 0;i<clusters.size();i++){
      for(int j = 0;j<clusters[i].data.size();j++){
        double temp = silhouette(clusters[i].data[j],i);
        sil += temp;
      }
    }
    sil /= (double)features.rows();
    cout << "Average silhouette score: " << sil << "\n";

		double separable = 0;
		for (int i = 0;i<clusters.size();i++){
			double tempMin = 100000000;
			for (int j = 0;j<clusters.size();j++){
				if (i != j){
						double temp = pow(calculateDistance(clusters[i].centroid,clusters[j].centroid),2);
						if (temp < tempMin){
							tempMin = temp;
						}
				}
			}
			separable += tempMin;
		}

		cout << "Separability score: " << separable << "\n";
}

double Clustering::calculateError(){
  double sum = 0;
  for (int i = 0;i<clusters.size();i++){
    double clusterSum = 0;
    for (int j = 0;j<clusters[i].data.size();j++){
      clusterSum += pow(calculateDistance(clusters[i].data[j],clusters[i].centroid),2);
    }
    cout << "SSE for cluster " << i << " is:" << clusterSum << "\n";
    sum += clusterSum;
  }
  return sum;
}

double Clustering::silhouette(vector<double>& ins,int cID){
  double tempA = aScore(ins,cID);
  double tempB = bScore(ins,cID);
  double tempC = 0;

  if (tempA > tempB){
    tempC = tempA;
  }
  else{
    tempC = tempB;
  }

  return (tempB-tempA)/tempC;
}

double Clustering::aScore(vector<double>& ins,int cID){
  size_t totalNum = clusters[cID].data.size();
  return calculateDistance(ins,clusters[cID].centroid)/totalNum;
}

double Clustering::bScore(vector<double>& ins,int cID){
  double highest = 10000000;
  for (int i = 0;i<clusters.size();i++){
    if (i != cID){
      double temp = aScore(ins,i);
      if (temp < highest){
        highest = temp;
      }
    }
  }
  return highest;
}

double Clustering::calculateDistance(vector<double>& input,vector<double>& testCase){
  double sum = 0;

  for (int i = 0;i<input.size();i++){
    if(attrValues[i] == 0){
      if(input[i] == UNKNOWN_VALUE || testCase[i] == UNKNOWN_VALUE){
        sum++;
      }
      else{
        sum += pow(input[i]-testCase[i],2); // euclidean distance
      }
    }
    else{
      if (input[i] != testCase[i] || input[i] == UNKNOWN_VALUE || testCase[i] == UNKNOWN_VALUE){
        sum++;
      }
    }
  }

  return sqrt(sum);
}
