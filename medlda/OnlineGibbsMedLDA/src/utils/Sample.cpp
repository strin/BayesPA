//
//  Sample.cpp
//  OnlineTopic
//
//  Created by Tianlin Shi on 5/1/13.
//  Copyright (c) 2013 Tianlin Shi. All rights reserved.
//

#include "Sample.h"

using namespace stl;

SampleWeight::SampleWeight(size_t K, double v) {
  eta_icov = vec2D<double>(K);
  eta_cov = vec2D<double>(K);
  prev_eta_icov = vec2D<double>(K);
  stat_icov = vec2D<double>(K);

  for(int k = 0; k < K; k++) {
    eta_icov[k].resize(K, 0);
    eta_icov[k][k] = 1 / (v * v);
    eta_cov[k].resize(K, 0);
    eta_cov[k][k] = v * v;
    prev_eta_icov[k].resize(K, 0);
    stat_icov[k].resize(K, 0);
  }

  eta_pmean = vec<double>(K, 0);
  eta_mean = vec<double>(K, 0);
  stat_pmean = vec<double>(K, 0);
  prev_eta_pmean = vec<double>(K, 0);

}

SampleZ::SampleZ(int D, stl::vec<int> W, size_t num_category) {
  Z = new int*[D];
  Zbar = new double*[D];
  Cdk = NULL;
  memset(Z, 0, sizeof(int*)*D);
  memset(Zbar, 0, sizeof(double*)*D);

  invlambda = new double*[num_category];
  for(size_t ci = 0; ci < num_category; ci++) {
    invlambda[ci] = new double[D];
  }

  for( int i = 0; i < D; i++) {
    Z[i] = new int[W[i]];
  }
  this->D = D;
  this->W = W;
  this->num_category = num_category;
}


SampleZ::~SampleZ() {
  if(Z != 0) {
    for( int i = 0; i < D; i++)
      if( Z[i] != 0)
        delete[] Z[i];
    delete[] Z;
  }

  if(Zbar != 0) {
    for( int i = 0; i < D; i++)
      if( Zbar[i] != 0)
        delete[] Zbar[i];
    delete[] Zbar;
  }

  if(invlambda != 0) {
    for(size_t ci = 0; ci < num_category; ci++) {
      delete[] invlambda[ci];
    }
    delete[] invlambda;
  }

  if(Cdk) {
    for( int i = 0; i < D; i++) {
      delete[] Cdk[i];
    }
    delete[] Cdk;
  }
}
