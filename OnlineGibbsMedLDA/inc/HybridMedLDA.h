//  HybridMedLDA is an EM-style inference algorithm for MedLDA.
//    it maximizes the lower bound of evidence.
//  E-Step: uses Gibbs sampling to compute Z and \lambda.
//  M-Step: uses variational inference to infer \eta (Gaussian) and \phi (Dirichlet).
//
//  Created by Tianlin Shi on 5/1/13.
//  Copyright (c) 2013 Tianlin Shi. All rights reserved.
//

#ifndef ___HybridMedLDA__
#define ___HybridMedLDA__

#include <iostream>

#include "utils.h"
#include "Sample.h"
#include "Corpus.h"
#include "InverseGaussian.h"
#include "MVGaussian.h"
#include "objcokus.h"

#include "stl.h"

typedef struct
{
  double time, ob_percent, accuracy;
  double* my;
}Commit;

class HybridMedLDA {
public:
  HybridMedLDA(int category = -1);
  ~HybridMedLDA();
  
  // draw samples from posterior.
  void updateZ(SampleZ* prevZ, CorpusData* dt);
  void updateLambda( SampleZ* prevZ, CorpusData* dt);
  void infer_Phi_Eta( SampleZ* prevZ, CorpusData* dt, bool reset);
  void normalize_Phi_Eta(int N, bool remove);
  void draw_Z_test(SampleZ* prevZ, int i, CorpusData* dt);
  void computeZbar(CorpusData* data, SampleZ* Z, int di);
  double computeDiscriFunc(CorpusData* dt, int di, Sample* sample, SampleZ* Z, double norm);
  
  // train the model.
  void init();
  double train(stl::vec2D<int> batch, stl::vec<int> label);
  vector<double> inference(vec2D<int> batch, int num_test_sample = -1, int category = -1);
  
  
  // input parameters.
  int K, T, I, J, J_burnin, category; // J is batch size.
  
  /* stats */
  stl::vec2D<double> gamma, prev_gamma;			  // sufficient statistics.
  stl::vec<double> gammasum;				  // sums of rows in gamma.
  stl::vec2D<double> eta_icov, eta_cov, prev_eta_icov;    // weight covariance matrix.
  stl::vec<double> eta_pmean, eta_mean, prev_eta_pmean;   // mean of eta without transformation.
  stl::vec2D<double> stat_phi, stat_icov;                 // stat used in global update.
  stl::vec<double> stat_pmean;                            // stat used in global update.
  vector<vector<double> > Zbar_test;

  stl::vec<int> stat_phi_list_k, stat_phi_list_t;		  // aux stats for sparse updates.
  stl::vec<int> prev_gamma_list_k, prev_gamma_list_t;         // aux stats for sparse updates.
  
  // experiment parameters.
  double alpha0, beta0, train_time;
  double c, l, v; // v: prior of eta.

  int maxSampleN, maxBurninN, max_gibbs_iter, testBurninN;

  /* source of randomness */
  InverseGaussian* invgSampler;
  MVGaussian* mvGaussianSampler;
  objcokus cokus;
};


#endif /* defined(__OnlineTopic__GibbsSampler__) */
