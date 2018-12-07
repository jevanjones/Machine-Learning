#ifndef DECISIONTREE_H
#define DECISIONTREE_H

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

class DecisionTree : public SupervisedLearner{
private:
  Rand& random;
  class Leaf{
  public:
    string attribute;
    int attrIndex;
    size_t values;
    size_t outValues;
    double majorityClass;
    bool isBottom;
    vector<Leaf> children;
    vector<vector<double> > data;
    vector<double> output;
    vector<string> attrNames;
    vector<size_t> attrValues;

    Leaf(){
      isBottom = 0;
    }
    ~Leaf(){}

    void clearAll(){ // reset the leaf
      attribute.clear();
      attrIndex = 0;
      values = 0;
      outValues = 0;
      majorityClass = 0;
      isBottom = 0;
      children.clear();
      data.clear();
      output.clear();
      attrNames.clear();
      attrValues.clear();
    }

    void printAll(){ // print the important info of the tree.
      cout << "Attribute: " << attribute << " with " << children.size() << " children\n";
      if (!children.empty() && !isBottom){
        for (int i = 0;i<children.size();i++){
            cout << "   Child Node " << i << ":\n";
            children[i].printAll();
        }
        cout << "\n";
      }
      else{
        cout << "Leaf Node\n";
      }
      cout << "Node " << attribute << " Finished\n";
    }

    void countNodes(int& counter){
      if (children.empty() || isBottom){
        return;
      }
      for (int i = 0;i<children.size();i++){
        counter++;
        children[i].countNodes(counter);
      }
    }

    void findDepth(int& depth,int currDepth){
      if (children.empty() || isBottom){
        return;
      }
      for (int i = 0;i<children.size();i++){
        if (currDepth > depth){
          depth = currDepth;
        }
        children[i].findDepth(depth,currDepth+1);
      }
    }

    void split(size_t attr){
      attrIndex = attr;
      attribute = attrNames[attr];
      values = attrValues[attr];
      for (size_t i = 0;i<attrValues[attr];i++){
          Leaf temp;
          temp.outValues = outValues;
          temp.values = 0;
          children.push_back(temp);
      }

      vector<string> nameTmp;
      nameTmp = attrNames;
      nameTmp.erase(nameTmp.begin()+attr);

      vector<size_t> valTmp;
      valTmp = attrValues;
      valTmp.erase(valTmp.begin()+attr);
      for (int i = 0;i<children.size();i++){
        children[i].attrNames = nameTmp; // save the useful names and values of everything post split in the children
        children[i].attrValues = valTmp;
      }

      for (int i = 0;i<data.size();i++){
        double index;
        index = data[i][attr];
        vector<double> temp;
        temp = data[i];
        temp.erase(temp.begin()+attr); //clip off the attribute we're splitting on
        children[index].data.push_back(temp); // save the row of data into the child
        children[index].output.push_back(output[i]); // save the output of that particular row
      }
    }

    double predictMajority(){
      double highest = 0;
      double value = 0;
        for (double i = 0;i<(double)outValues;i++){
          double temp = std::count(output.begin(),output.end(),i); // count the values in the output to find the majority class
          if (temp > highest){
            highest = highest;
            value = i;
          }
        }
        majorityClass = value;
        return value;
    }

    double predictAll(const std::vector<double>& feat,map<string,int> ind){
      if (!children.empty() && !isBottom){
        return children[feat[ind[attribute]]].predictAll(feat,ind);
      }
      return predictMajority();
    }
  };

  Leaf tree;
  map<string,int> attr_to_index;
  map<string,size_t> attr_to_value;

  size_t calculateSplitAttr(Leaf leaf);
  void induceCompleteTree(Leaf& leaf);
  void pruneTree(Leaf& leaf,Matrix& features, Matrix& labels);
  double calculateAccuracy(Matrix& features,Matrix& labels);
public:
  DecisionTree(Rand& r) : SupervisedLearner(), random(r){}
  ~DecisionTree() {}

  void train (Matrix& features, Matrix& labels);

  void predict (const std::vector<double>& features,std::vector<double>& labels);
};
#endif
