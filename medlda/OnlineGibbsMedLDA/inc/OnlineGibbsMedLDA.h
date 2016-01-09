//  OnlineGibbsMedLDA is a mean-field inference algorithm for MedLDA.
//    it maximizes the lower bound of evidence.
//  E-Step: uses Gibbs sampling to compute Z and \lambda.
//  M-Step: uses variational inference to infer \eta (Gaussian) and \phi (Dirichlet).
//
//  Created by Tianlin Shi on 5/1/13.
//  Copyright (c) 2013 Tianlin Shi. All rights reserved.
//

#ifndef ___OnlineGibbsMedLDA__
#define ___OnlineGibbsMedLDA__

#include <iostream>

#include "utils.h"
#include "Sample.h"
#include "Corpus.h"
#include "InverseGaussian.h"
#include "MVGaussian.h"
#include "objcokus.h"

#include "stl.h"

class OnlineGibbsMedLDA {
public:
  OnlineGibbsMedLDA(const vector<int>& category);
  void init();

  ~OnlineGibbsMedLDA();
  
  /* sample local variables */
  void updateZ(stl::ptr<SampleZ> prevZ, stl::ptr<CorpusData> dt);
  void draw_Z_test(ptr<SampleZ> prevZ, int i, stl::ptr<CorpusData> dt, vec2D<double>& Ckt, vec<double>& Ckt_sum);

  void computeZbar(stl::ptr<CorpusData> data, ptr<SampleZ> Z, int di);

  void updateLambda( ptr<SampleZ> prevZ, stl::ptr<CorpusData> dt);
  
  /* update global stats */
  void infer_Phi_Eta( ptr<SampleZ> prevZ, stl::ptr<CorpusData> dt, bool reset);
  void normalize_Phi_Eta(int N, bool remove);

  
  /* train the model */
  // deal with single label.
  double train(const stl::vec2D<int>& batch, const stl::vec<int>& label);
  // deal with multiple labels.
  double train(const stl::vec2D<int>& batch, const stl::vec2D<int>& label);
  vector<double> inference(vec2D<int> batch, int num_test_sample = -1);
  
  /* hyper parameters */
  int K;	    // number of topics.
  int T;	    // number of total words.
  int num_category; // number of categories.
  vector<int> category; // the categories this model tries to learn.
  int I;            // number of mean-field rounds for each BayesPA update.
  int J;            // number of Gibbs samples in the mean-field update of latent variables (substract J_burnin)
  int J_burnin;     // number of burn-in steps for Gibbs samples of latent variables.
  double alpha0;    // prior of document topic distribution.
  double beta0;     // prior of dictionary.
  double c;         // regularization parameter of hinge-loss.
  double l;         // margin parameter of hinge-loss.
  double v;         // prior weight \sim N(0, v^2).
  double stepsize;  // weight for each data point.
  bool point_estimate_for_test; // whether use point estimates at test time.
  
  /* stats */
  stl::vec2D<double> gamma, prev_gamma;			  // sufficient statistics.
  stl::vec<double> gammasum;				  // sums of rows in gamma.
  stl::vec2D<double> eta_icov, eta_cov, prev_eta_icov;    // weight covariance matrix.
  stl::vec<double> eta_pmean, eta_mean, prev_eta_pmean;   // mean of eta without transformation.
  stl::vec2D<double> stat_phi, stat_icov;                 // stat used in global update.
  stl::vec<double> stat_pmean;                            // stat used in global update.

  stl::vec<int> stat_phi_list_k, stat_phi_list_t;		  // aux stats for sparse updates.
  stl::vec<int> prev_gamma_list_k, prev_gamma_list_t;         // aux stats for sparse updates.
  
  /* results */
  double train_time;                      // training time spent.
  vector<vector<double> > Zbar_test;      // emprirical topic distribution of last inferred corpus.

  /* source of randomness */
  InverseGaussian* invgSampler;
  MVGaussian* mvGaussianSampler;
  objcokus cokus;
};


#endif /* defined(__OnlineTopic__GibbsSampler__) */
