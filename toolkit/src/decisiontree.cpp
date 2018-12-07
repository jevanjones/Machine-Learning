//-------------------------------//
// Decision Tree Calss Functions //
//-------------------------------//

#include "decisiontree.h"
#include <iostream>
#include <stdio.h>
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>
#include <math.h>
#include "error.h"
#include "rand.h"

using std::vector;
using std::cout;

void printVector(vector<double> v){
  for (int i = 0;i<v.size();i++){
    cout << v[i] << ",";
  }
  cout << "\n";
}

void printVectorString(vector<string> v){
  for (int i = 0;i<v.size();i++){
    cout << v[i] << ",";
  }
  cout << "\n";
}

void printVectorVal(vector<size_t> v){
  for (int i = 0;i<v.size();i++){
    cout << v[i] << ",";
  }
  cout << "\n";
}

void printDVector(vector<vector<double> > v){
  for (int i = 0;i<v.size();i++){
    printVector(v[i]);
  }
  cout << "\n";
}

vector<vector<double> > transpose(vector<vector<double> > d){ // take the transpose of the given vector
  vector<vector<double> > retVal;
  for (int i = 0;i<d[0].size();i++){
    vector<double> temp;
    for (int j = 0;j<d.size();j++){
      temp.push_back(d[j][i]);
    }
    retVal.push_back(temp);
  }

  return retVal;
}

size_t DecisionTree::calculateSplitAttr(Leaf leaf){ // given the ID3 splitting algorithm of standard gain, that's what this function calculates
    vector<vector<double> > trData;
    trData = transpose(leaf.data);
    size_t splitAttr = 0; // the attribute that will be split on. Also the return value
    double highestGain = 0; // this catches the highest gain found of all the attributes that can be split on

    //cout << "Finding Info Gain...\n";

//    cout << "leaf output values:\n";
//    printVector(leaf.output);
//    cout << "leaf output names:\n";
//    printVectorString(leaf.attrNames);
//    cout << "leaf attribute values:\n";
//    printVectorVal(leaf.attrValues);
//    cout << "leaf data:\n";
//    printDVector(leaf.data);
//    cout << "Transpose data:\n";
//    printDVector(trData);
//    cout << "Out values of leaf:" << leaf.outValues << "\n";

    double infoGain = 0; // find the gain of the output itself
    for (int i = 0;i<leaf.outValues;i++){
      double temp = (double)std::count(leaf.output.begin(),leaf.output.end(),i)/(double)leaf.output.size();
      if (temp != 0){
        temp = -temp*log2(temp);
      }
      infoGain += temp;
    }

    //cout << "Info gain is: " << infoGain << "\nFinding Attr Gain...\n";

    for (int i = 0;i<trData.size();i++){ // and next calculate the gain of the different attributes
      double attrGain = 0;
      for (int k = 0;k<leaf.attrValues[i];k++){
        double attrCount = (double)std::count(trData[i].begin(),trData[i].end(),k);
        double inner = 0;
        for (int j = 0;j<leaf.outValues;j++){
          double temp = 0;
          for (int l = 0;l<trData[i].size();l++){
            if (trData[i][l] == k && leaf.output[l] == j){
              temp++;
            }
          }
          if (temp == 0){
            inner += 0;
          }
          else{
            inner += -(temp/attrCount)*log2(temp/attrCount);
          }
        }
        attrGain += (attrCount/trData[i].size())*inner;
      }

      //cout << "Attr: " << leaf.attrNames[i] << " Gain: " << attrGain << "\n";
      double tmpGain = infoGain - attrGain;
      if (tmpGain > highestGain){ // save the highest gain
        highestGain = tmpGain;
        splitAttr = (size_t)i;
      }
    }

    //cout << "Will try to split on: " << splitAttr << "\n";

    return splitAttr;
}

void DecisionTree::induceCompleteTree(Leaf& leaf){
    if (leaf.attrNames.size() == 1){
      leaf.attribute = leaf.attrNames[0];
      //cout << "Can't split anymore, will just return now\n";
      return;
    }
    else if (std::count(leaf.output.begin(),leaf.output.end(),leaf.output[0]) == leaf.output.size()){
      //cout << "All one output class, won't split anymore\n";
      leaf.attribute = leaf.attrNames[0];
      return;
    }
    else{
      leaf.split(calculateSplitAttr(leaf));
      for (int i = 0;i<leaf.children.size();i++){ // If we can still split, we will continue to split on the children of this node
        //cout << "Going down another layer\n";
        induceCompleteTree(leaf.children[i]);
      }
    }
}

void DecisionTree::train(Matrix& features, Matrix& labels){

      //cout  << "Starting the training\n";
      Matrix trainingSet;
      Matrix trainingLabels;
      Matrix validationSet;
      Matrix validationLabels;

      attr_to_index.clear();
      attr_to_value.clear();

      tree.clearAll();

      features.shuffleRows(random,&labels); // I shuffle my features so that I don't simply memorize the dataset, but actually learn

      trainingSet.copyPart(features,0,0,features.rows()*0.9,features.cols()); // and create a validation set from the given dataset
      validationSet.copyPart(features,features.rows()*0.9,0,features.rows()-features.rows()*0.9,features.cols());
      trainingLabels.copyPart(labels,0,0,labels.rows()*0.9,labels.cols());
      validationLabels.copyPart(labels,labels.rows()*0.9,0,labels.rows()-labels.rows()*0.9,labels.cols());

      //cout << "setting up the data\n";

      for (size_t i = 0;i<trainingSet.rows();i++){ // save the dataset into a double vector for the leaves of the tree to use
        vector<double> temp;
        for (size_t j = 0;j<trainingSet.cols();j++){
          temp.push_back(trainingSet[i][j]);
        }
        tree.output.push_back(trainingLabels[i][0]);
        tree.data.push_back(temp);
      }

      tree.outValues = trainingLabels.valueCount(0);

      for (size_t j = 0;j<features.cols();j++){ // set up the metadata to be used by the tree
        attr_to_index[features.attrName(j)] = j;
        attr_to_value[features.attrName(j)] = features.valueCount(j);
        tree.attrNames.push_back(features.attrName(j));
        tree.attrValues.push_back(features.valueCount(j));
      }

      //cout << "Inducing the tree\n";
      induceCompleteTree(tree);

      int treeDepth = 0;
      int nodeCount = 0;

      tree.countNodes(nodeCount);
      tree.findDepth(treeDepth,0);

//      cout << "Before pruning:\nNum of nodes in the tree: " << nodeCount << "\nMax Tree Depth: " << treeDepth << "\n";
//      cout << "Validation Set accuracy: " << calculateAccuracy(validationSet,validationLabels) << "\n";
      pruneTree(tree,validationSet,validationLabels);

      treeDepth = 0;
      nodeCount = 0;

      tree.countNodes(nodeCount);
      tree.findDepth(treeDepth,0);

//      cout << "After pruning:\nNum of nodes in the tree: " << nodeCount << "\nMax Tree Depth: " << treeDepth << "\n";
//      cout << "Validation Set accuracy: " << calculateAccuracy(validationSet,validationLabels) << "\n";
      //tree.printAll();
}

void DecisionTree::pruneTree(Leaf& leaf,Matrix& features, Matrix& labels){ // this prunes the tree by checking accuracies
    if (leaf.children.empty()){
      return;
    }
    double prevAccuracy = calculateAccuracy(features,labels);

    leaf.isBottom = 1;
    double newAccuracy = calculateAccuracy(features,labels);

    if ((prevAccuracy - newAccuracy) > 0.001){
      leaf.isBottom = 0;
      for (int i = 0;i<leaf.children.size();i++){
        pruneTree(leaf.children[i],features,labels);
      }
    }
}

double DecisionTree::calculateAccuracy(Matrix& features,Matrix& labels){ // This calculates the accuracy of the tree given the dataset
    double accuracy = 0;
    for (size_t k = 0;k<features.rows();k++){
      if (labels[k][0] == tree.predictAll(features[k],attr_to_index)){
        accuracy++;
      }
    }
    accuracy /= (double)features.rows();
    return accuracy;
}

void DecisionTree::predict(const std::vector<double>& features,std::vector<double>& labels){
      //cout << "Predicting...\n";
      labels[0] = tree.predictAll(features,attr_to_index);
}
