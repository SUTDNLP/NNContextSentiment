/*
 * SparseClassifier.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_SparseClassifier_H_
#define SRC_SparseClassifier_H_

#include <iostream>

#include <assert.h>
#include "Example.h"
#include "Feature.h"

#include "SparseUniLayer.h"
#include "N3L.h"

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

//A native neural network classfier using only word embeddings
template<typename xpu>
class SparseClassifier {
public:
  SparseClassifier() {
    _dropOut = 0.5;
  }
  ~SparseClassifier() {

  }

public:
  int _labelSize;
  int _linearfeatSize;

  dtype _dropOut;
  Metric _eval;

  SparseUniLayer<xpu> _layer_linear;

public:

  inline void init(int labelSize, int linearfeatSize) {
    _labelSize = labelSize;
    _linearfeatSize = linearfeatSize;

    _layer_linear.initial(_labelSize, _linearfeatSize, false, 4, 2);
    _eval.reset();

  }

  inline void release() {
    _layer_linear.release();
  }


  inline dtype process(const vector<Example>& examples, int iter) {
    _eval.reset();

    int example_num = examples.size();
    dtype cost = 0.0;
    int offset = 0;
    for (int count = 0; count < example_num; count++) {
      const Example& example = examples[count];

      Tensor<xpu, 2, dtype> output, outputLoss;

      output = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);
      outputLoss = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);

      vector<int> m_linears;

      //forward propagation
      //input setting, and linear setting
      srand(iter * example_num + count );
      m_linears.clear();
      for (int idy = 0; idy < example.m_linears.size(); idy++) {
        if (1.0 * rand() / RAND_MAX >= _dropOut) {
          m_linears.push_back(example.m_linears[idy]);
        }
      }
      _layer_linear.ComputeForwardScore(m_linears, output);


      cost += softmax_loss(output, example.m_labels, outputLoss, _eval, example_num);

      // loss backward propagation
      _layer_linear.ComputeBackwardLoss(m_linears, output, outputLoss);

      //release
      FreeSpace(&output);
      FreeSpace(&outputLoss);    }

    if (_eval.getAccuracy() < 0) {
      std::cout << "strange" << std::endl;
    }

    return cost;
  }

  int predict(const vector<int>& linears, const vector<Feature>& features, vector<dtype>& results) {
    Tensor<xpu, 2, dtype> output;

    output = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);

    _layer_linear.ComputeForwardScore(linears, output);

    // decode algorithm
    int optLabel = softmax_predict(output, results);

    //release
    FreeSpace(&output);

    return optLabel;
  }

  dtype computeScore(const Example& example) {
    Tensor<xpu, 2, dtype> output;

    output = NewTensor<xpu>(Shape2(1, _labelSize), d_zero);

    _layer_linear.ComputeForwardScore(example.m_linears, output);

    dtype cost = softmax_cost(output, example.m_labels);

    //release
    FreeSpace(&output);

    return cost;
  }

  void updateParams(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
    _layer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);
  }

  void writeModel();

  void loadModel();

  void checkgrads(const vector<Example>& examples, int iter) {
    checkgrad(this, examples, _layer_linear._W, _layer_linear._gradW, "_layer_linear._W", iter, _layer_linear._indexers, false);
    checkgrad(this, examples, _layer_linear._b, _layer_linear._gradb, "_layer_linear._b", iter);
  }

public:
  inline void resetEval() {
    _eval.reset();
  }

  inline void setDropValue(dtype dropOut) {
    _dropOut = dropOut;
  }

};

#endif /* SRC_SparseClassifier_H_ */
