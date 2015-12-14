/*
 * Example.h
 *
 *  Created on: Mar 17, 2015
 *      Author: mszhang
 */

#ifndef SRC_EXAMPLE_H_
#define SRC_EXAMPLE_H_

#include "Feature.h"

using namespace std;

class Example {

public:
  vector<int> m_labels;
  vector<Feature> m_features;
  vector<int> m_linears;

public:
  Example()
  {

  }
  virtual ~Example()
  {

  }

  void clear()
  {
    m_labels.clear();
    m_features.clear();
    m_linears.clear();
  }


};

#endif /* SRC_EXAMPLE_H_ */
