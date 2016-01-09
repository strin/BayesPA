//
//  Sample.h
//  OnlineTopic
//
//  Created by Tianlin Shi on 5/1/13.
//  Copyright (c) 2013 Tianlin Shi. All rights reserved.
//
/*
 * Sample Representation of Document Corpus.
 */

#ifndef __OnlineTopic__Sample__
#define __OnlineTopic__Sample__

#include <iostream>
#include "utils.h"
#include "stl.h"

/* samples of global parameters.
 */
class SampleWeight {
public:
  SampleWeight(size_t K, double v);

  stl::vec2D<double> eta_icov, eta_cov, prev_eta_icov;    // weight covariance matrix.
  stl::vec<double> eta_pmean, eta_mean, prev_eta_pmean;   // mean of eta without transformation.
  stl::vec2D<double> stat_icov;                 // stat used in global update.
  stl::vec<double> stat_pmean;                            // stat used in global update.
};


/* samples of local latent variables.
 */
class SampleZ {
public:
  // construction & destruction.
  SampleZ( int D, stl::vec<int> W, size_t num_category);
  ~SampleZ();
  // parameters.
  int D;
  stl::vec<int> W;
  size_t num_category;
  // content.
  int** Z; // samples of hidden units.
  double** Cdk;
  double** Zbar;
  double** invlambda; // data augmentation.
};
#endif /* defined(__OnlineTopic__Sample__) */
