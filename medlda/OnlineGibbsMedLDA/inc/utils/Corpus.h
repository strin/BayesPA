//
//  Corpus.h
//  OnlineTopic
//
//  Created by Tianlin Shi on 4/29/13.
//  Copyright (c) 2013 Tianlin Shi. All rights reserved.
//

#ifndef __OnlineTopic__Corpus__
#define __OnlineTopic__Corpus__

#include <iostream>
#include <string>
#include <functional>
#include <cassert>
#include <dirent.h>

#include "debug.h"
#include "utils.h"
#include "stl.h"


using namespace std;
using namespace stl;


class DataSample {

public:

  DataSample() {}

  /* create data sample for multi-class classification */
  DataSample(const vector<int>& ex, const int& label) {
    this->W = ex.size();
    if(this->W == 0) return;
    this->words.resize(this->W);
    for(int wi = 0; wi < this->W; wi++) this->words[wi] = ex[wi];
    this->labels.push_back(label);
  }

  /* create data sample for multi-class classification */
  DataSample(const vector<int>& ex, const vector<int>& labels) {
    this->W = ex.size();
    if(this->W == 0) return;
    this->words.resize(this->W);
    for(int wi = 0; wi < this->W; wi++) this->words[wi] = ex[wi];
    this->labels = labels;
  }

  vec<int> words;  // the words in the document.
  vec<int> labels, pred;  // the ground truth label and predicted label.
  int W;       // number of words.
};


/* transform dataset into form that MedLDA processes.
 * for num_cateory, a y-vector is created. fro exampke, 
 * if cat = [3,4] and num_category 5, then y =[-1,-1,-1,1,1].
 */
class CorpusData {
public:
  CorpusData() {}
  
  CorpusData(const vec2D<int>& docs, size_t num_category) {
    const size_t data_size = docs.size();
    vec<shared_ptr<DataSample> > data(data_size);

    for(int di = 0; di < data_size; di++) {
      vector<int> label;
      data[di] = make_shared<DataSample>(docs[di], label);
    }
    
    this->init(data, num_category);
  }

  CorpusData(const vec2D<int>& docs, const vec2D<int>& labels, size_t num_category) {
    const size_t data_size = docs.size();
    vec<shared_ptr<DataSample> > data(data_size);

    for(int di = 0; di < data_size; di++) {
      data[di] = make_shared<DataSample>(docs[di], labels[di]);
    }
    
    this->init(data, num_category);
  }


  CorpusData(const vec<shared_ptr<DataSample> >& data, size_t num_category) {
    this->init(data, num_category);
  }


  void init(const vec<shared_ptr<DataSample> >& data, size_t num_category) {
    // (TODO) replace the manual memory management with shared_ptr.
    this->D = data.size();
    this->W.resize(this->D);
    this->data.resize(this->D);
    this->y.resize(this->D);
    this->py.resize(this->D);
    this->my.resize(this->D);

    for(int i = 0; i < this->D; i++) {
      auto data_sample = data[i];
      this->W[i] = data_sample->W;
      this->y[i].resize(num_category, -1);

      for(auto label: data_sample->labels) {
        assert(label < num_category && label >= 0);
        this->y[i][label] = 1;
      }

      this->py[i].resize(num_category, -1);
      this->my[i].resize(num_category, 0);
      
      this->data[i].resize(this->W[i]);
      for(int w = 0; w < this->W[i]; w++) this->data[i][w] = data_sample->words[w];
    }

  }

  int D;
  vec<int> W;       // length of each document.
  vec2D<int> data;  // document words.
  vec2D<int> y, py; // document y-vectors and predicted y-vectors.
  vec2D<double> my; // confidence level (margin).
};

#endif /* defined(__OnlineTopic__Corpus__) */
