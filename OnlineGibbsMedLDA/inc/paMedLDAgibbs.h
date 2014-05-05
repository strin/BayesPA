#ifndef ___paMedLDA_gibbs___
#define ___paMedLDA_gibbs___

#include <iostream>

#include "utils.h"
#include "Sample.h"
#include "Corpus.h"
#include "InverseGaussian.h"
#include "MVGaussian.h"
#include "objcokus.h"

typedef struct
{
	double time, ob_percent, accuracy;
	double* my;
}Commit;

class paMedLDAgibbs {
public:
	paMedLDAgibbs( Corpus* corpus, int category = 1);
	~paMedLDAgibbs();
	
	// draw samples from posterior.
	void updateZ( SampleZ* prevZ, int batchIdx, int batchSize);
	void updateLambda( SampleZ* prevZ, int batchIdx, int batchSize);
	void infer_Phi_Eta( SampleZ* prevZ, int batchIdx, int batchSize, bool reset);
	void normalize_Phi_Eta(int N, bool remove);
	void draw_Z_test(Sample* sample, SampleZ* prevZ, int i, CorpusData* dt);
	void computeZbar(CorpusData* data, SampleZ* Z, int batchIdx);
	double computeDiscriFunc(CorpusData* dt, int di, Sample* sample, SampleZ* Z, double norm);
	int eject_sample(CorpusData* dt, SampleZ* Z, int someround); // cancel the effect of mini-batch at some round.
	
	// train the model.
	void init();
	double train(int num_iter);
	double inference(CorpusData* testData, int num_test_sample);
	
	// output parameters = samples.
	deque<Sample*>* samples;
	deque<SampleZ*>* sampleZs;
	
	// input parameters.
	int K, T, I, J, J_burnin, category; // J is batch size.
	Corpus* corpus;
	
	// training and testing data.
	CorpusData *train_data, *test_data;
	double *my, *py;
	
	// training data param for convenience.
	int batchSize, round; // round is the clock of online algo.
	bool lets_batch, lets_multic; // batch mode.
	
	// stats.
	double** gamma, **prev_gamma; // emperical sum(Od(k,t))
	double* gammasum; // row sum of gamma.
	double** eta_icov, **eta_cov, **prev_eta_icov;
	double* eta_pmean, *eta_mean, *prev_eta_pmean; // mean of eta without transformation.
	double **Cdk_test, **Zbar_test, **Ckt_test;
	double *Ckt_test_sum;
	double **stat_phi, **stat_icov, *stat_pmean; // stat used in global update.
	int *stat_phi_list_k, *stat_phi_list_t, stat_phi_list_end; // aux stat for sparse update.
	int *prev_gamma_list_k, *prev_gamma_list_t, prev_gamma_list_end;
	double *forget_factor; // cumulative forget factor for each round.
	SampleZ* iZ_test, *iZ;
	
	// experiment parameters.
	double alpha0, beta0, train_time;
	double c, l, v; // v: prior of eta.
	deque<int> *batchAlive;
	int maxSampleN, maxBurninN, max_gibbs_iter, testBurninN;
	int epoch, batchIdx;
	double pos_ratio; // ratio of positive examples in the training set.
	double u, tao; // gradient contribution.
	bool mode_additive; // use sequential bayesian experiment mode.
	// aux samplers.
	InverseGaussian* invgSampler;
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
