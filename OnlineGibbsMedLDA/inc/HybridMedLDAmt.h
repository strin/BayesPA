//  HybridMedLDA_MT is an EM-style inference algorithm for MedLDA.
//		it maximizes the lower bound of evidence.
//  E-Step: uses Gibbs sampling to compute Z and \lambda.
//  M-Step: uses variational inference to infer \eta (Gaussian) and \phi (Dirichlet).
//
//  Created by Tianlin Shi on 5/1/13.
//  Copyright (c) 2013 Tianlin Shi. All rights reserved.
//

#ifndef ___HybridMedLDA__MT__
#define ___HybridMedLDA__MT__

#include <iostream>

#include "utils.h"
#include "Sample.h"
#include "Corpus.h"
#include "InverseGaussian.h"
#include "MVGaussian.h"
#include "objcokus.h"
#include "HybridMedLDA.h"

class WeightStat {
public:
	double** eta_icov, **eta_cov;
	double* eta_pmean, *eta_mean; // mean of eta without transformation.
	double **stat_icov, *stat_pmean;
	double *invlambda;
};

class HybridMedLDAmt {
public:
	HybridMedLDAmt( Corpus* corpus, int category_n);
	~HybridMedLDAmt();
	
	// draw samples from posterior.
	void updateZ( SampleZ* prevZ, CorpusDataMt* dt, int batchIdx, int batchSize);
	void updateLambda( SampleZ* prevZ, CorpusDataMt* dt, int batchIdx, int batchSize);
	void infer_Phi_Eta( SampleZ* prevZ, CorpusDataMt* dt, bool reset, int batchIdx, int batchSize);
	void normalize_Phi_Eta(int N, bool remove);
	void draw_Z_test(Sample* sample, SampleZ* prevZ, int i, CorpusDataMt* dt);
	void computeZbar(CorpusDataMt* data, SampleZ* Z, int batchIdx);
	double computeDiscriFunc(CorpusDataMt* dt, int di, Sample* sample, SampleZ* Z, double norm);
	int eject_sample(CorpusDataMt* dt, SampleZ* Z, int someround); // cancel the effect of mini-batch at some round.
	
	// train the model.
	void init();
	double train();
	double inference(CorpusDataMt* testData);
	
	// input parameters.
	int K, T, I, J, J_burnin, category_n; // J is batch size.
	Corpus* corpus;
	
	// training and testing data.
	CorpusDataMt *data, *testData;
	
	// training data param for convenience.
	int batchSize, round; // round is the clock of online algo.
	
	// stats.
	double** gamma; // emperical sum(Od(k,t))
	double* gammasum; // row sum of gamma.
	double **Cdk_test, **Zbar_test, **Ckt_test;
	double *Ckt_test_sum;
	double **stat_phi; // stat used in global update.
	SampleZ* iZ_test;
	WeightStat** wstat;
	int *stat_phi_list_k, *stat_phi_list_t, stat_phi_list_end; // aux stat for sparse update.
	
	// experiment parameters.
	double alpha0, beta0, train_time;
	double m_c, m_l, m_v; // v: prior of eta.
	deque<int> *batchAlive;
	int maxSampleN, maxBurninN, max_gibbs_iter, testBurninN;
	int epoch;
	double u, tao; // gradient contribution.
	bool mode_additive; // use sequential bayesian experiment mode.
	
	// aux samplers.
	InverseGaussian** invgSampler;
	MVGaussian* mvGaussianSampler;
	objcokus cokus;
	
	// result.
	double test_acc;
	vector<Commit> commit_points;
	bool lets_commit;
	int commit_point_spacing, commit_point_n;
private:
};


#endif /* defined(__OnlineTopic__GibbsSampler__) */
