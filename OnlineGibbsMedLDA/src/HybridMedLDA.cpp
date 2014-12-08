
//  GibbsSampler.cpp
//  OnlineTopic
//
//  Created by Tianlin Shi on 5/1/13.
//  Copyright (c) 2013 Tianlin Shi. All rights reserved.
//

#include "HybridMedLDA.h"
#include "utils.h"
#include "gammaln.h"

using namespace stl;
using namespace std;

HybridMedLDA::HybridMedLDA(Corpus* corpus, int category) {
  this->corpus = corpus;
  this->category = category;
  this->num_category = corpus->newsgroupN;
  samples = new deque<Sample*>();
  sampleZs = new deque<SampleZ*>();
  invgSampler = new InverseGaussian();
  mvGaussianSampler = new MVGaussian();
  data = new CorpusData();
  testData = new CorpusData();
  
  // set basic parameters.
  T                  = (int)corpus->T;
  data->D                = (int)corpus->trainDataSize;
  testData->D              = (int)corpus->testDataSize;
  // key parameters.
  K                  = 5; // number of topics.
  batchSize              = data->D;   // mini-batch size.
  epoch                = 1;     // number of scans of the entire corpus.
  I                  = 2;
  J                  = 2;   // Z repeated sample size.
  u                  = 1;
  tao                  = 1;
  J_burnin              = 0;
  l                  = 164;
  max_gibbs_iter            = 50;
  testBurninN              = 10;
  alpha0                = 1/(double)K;  // prior of document topic distribution.
  beta0                = 0.5;      // prior of dictionary.

  // other parameters.
  v                  = 1;
  c                  = 1;
  lets_commit              = false;
  lets_batch              = false;
  lets_multic              = false;
  mode_additive            = false;
  commit_point_spacing        = 1;
  
  cokus.reloadMT();
  cokus.seedMT(time(NULL)+category);
}

void HybridMedLDA::init() {
  /* sampling setting */
  maxSampleN              = 1;
  if(lets_commit) {
    maxBurninN            = commit_point_spacing*commit_point_n;
  }else{
    maxBurninN            = epoch*data->D/batchSize+1;
  }
    
  /* data init*/
  data->W                = new int[data->D];
  testData->W              = new int[testData->D];
  data->data              = new int*[data->D];
  data->y                = new int[data->D];
  data->py              = new int[data->D];
  data->my              = new double[data->D];
  testData->data            = new int*[testData->D];
  testData->y              = new int[testData->D];
  testData->py            = new int[testData->D];
  testData->my            = new double[testData->D];
  int* idx = new int[data->D];
  for(int i = 0; i < data->D; i++) idx[i] = i;
  shuffleArray<int>(idx, data->D, cokus, data->D);
  int train_pos = 0, train_neg = 0, test_pos = 0, test_neg = 0;
  
  for(int i = 0; i < data->D; i++) {
    data->W[i] = corpus->trainData[idx[i]]->W;
    if(category == -1)
      data->y[i] = corpus->trainData[idx[i]]->label;
    else{
      if(corpus->multi_label) {
        data->y[i] = -1;
        for(int j = 0; j < corpus->trainData[idx[i]]->multi_label.size(); j++) {
          if(category == corpus->trainData[idx[i]]->multi_label[j])
            data->y[i] = 1;
        }
      }else
        data->y[i] = corpus->trainData[idx[i]]->label == category ? 1 : -1;
    }
    train_neg += (1-data->y[i])/2;
    train_pos += (1+data->y[i])/2;
    data->py[i] = 0;
    data->data[i] = new int[data->W[i]];
    for(int w = 0; w < data->W[i]; w++) data->data[i][w] = corpus->trainData[idx[i]]->words[w];
  }
  for(int i = 0; i < testData->D; i++) {
    testData->W[i] = corpus->testData[i]->W;
    if(category == -1)
      testData->y[i] = corpus->testData[i]->label;
    else{
      if(corpus->multi_label) {
        testData->y[i] = -1;
        for(int j = 0; j < corpus->testData[i]->multi_label.size(); j++) {
          if(category == corpus->testData[i]->multi_label[j])
            testData->y[i] = 1;
        }
      }else
        testData->y[i] = corpus->testData[i]->label == category ? 1 : -1;
    }
    test_neg += (1-testData->y[i])/2;
    test_pos += (1+testData->y[i])/2;
    testData->py[i] = 0;
    testData->data[i] = new int[testData->W[i]];
    for(int w = 0; w < testData->W[i]; w++) testData->data[i][w] = corpus->testData[i]->words[w];
  }
  
  /* stat init*/
  eta_icov = new double*[K];
  eta_cov = new double*[K];
  prev_eta_icov = new double*[K];
  eta_pmean = new double[K];
  eta_mean = new double[K];
  stat_pmean = new double[K];
  prev_eta_pmean = new double[K];
  memset(stat_pmean, 0, sizeof(double)*K);
  memset(eta_pmean, 0, sizeof(double)*K);
  memset(eta_mean, 0, sizeof(double)*K);
  stat_icov = new double*[K];
  for(int k = 0; k < K; k++) {
    prev_eta_icov[k] = new double[K];
    eta_cov[k] = new double[K];
    eta_icov[k] = new double[K];
    memset(eta_icov[k], 0, sizeof(double)*K);
    eta_icov[k][k] = 1/(v*v);
    stat_icov[k] = new double[K];
    memset(stat_icov[k], 0, sizeof(double)*K);
  }
  gamma = new double*[K];
  prev_gamma = new double*[K];
  gammasum = new double[K];
  for(int k = 0; k < K; k++) {
    gamma[k] = new double[T];
    prev_gamma[k] = new double[T];
  }
  stat_phi = new double*[K];
  prev_gamma_list_k = new int[K*T];
  prev_gamma_list_t = new int[K*T];
  prev_gamma_list_end = -1;
  stat_phi_list_k = new int[K*T];
  stat_phi_list_t = new int[K*T];
  stat_phi_list_end = -1;
  for(int i = 0; i < K; i++) {
    stat_phi[i] = new double[T];
    memset(stat_phi[i], 0, sizeof(double)*T);
  }
  pos_ratio = (double)train_pos/(double)(train_pos+train_neg);
  
  commit_points.clear();
  /* cleaning */
  delete[] idx;
  
  train_time = 0;
  /* Initialization */
  invgSampler->reset(1, 1);
  for(int k1 = 0; k1 < K; k1++) {
    for(int k2 = 0; k2 < K; k2++) {
      if(k1 == k2) {
        eta_icov[k1][k2] = 1/v/v;
        eta_cov[k1][k2] = v*v;
      }else{
        eta_icov[k1][k2] = eta_cov[k1][k2] = 0;
      }
    }
  }
  memset(eta_mean, 0, sizeof(double)*K);
  for(int k = 0; k < K; k++) {
    gammasum[k] = 0;
    for(int t = 0; t < T; t++) {
      gamma[k][t] = 0;
    }
  }
  
  batchIdx = 0;
}

HybridMedLDA::~HybridMedLDA() {
  if(!data->W) return; // not initialized.
  
  /* clean data */
  delete[] data->W;
  delete[] data->y;
  delete[] data->py;
  delete[] data->my;
  for(int i = 0; i < data->D; i++) {
    delete[] data->data[i];
  }
  for(int i = 0; i < testData->D; i++) {
    delete[] testData->data[i];
  }
  delete[] data->data;
  delete[] testData->W;
  delete[] testData->y;
  delete[] testData->py;
  delete[] testData->my;
  delete[] testData->data;
  
  /* clean stat */
  for(int i = 0; i < K; i++) {
    delete[] gamma[i];
    delete[] prev_gamma[i];
    delete[] eta_icov[i];
    delete[] eta_cov[i];
    delete[] prev_eta_icov[i];
    delete[] stat_icov[i];
    delete[] stat_phi[i];
  }
  delete[] stat_phi;
  delete[] stat_phi_list_k;
  delete[] stat_phi_list_t;
  delete[] gamma;
  delete[] prev_gamma;
  delete[] gammasum;
  delete[] eta_icov;
  delete[] eta_cov;
  delete[] stat_icov;
  delete[] stat_pmean;
  delete[] eta_pmean;
  delete[] eta_mean;
  delete[] prev_eta_pmean;
  delete[] prev_eta_icov;
  for(int i = 0; i < samples->size(); i++) {
    Sample* sample = samples->back();
    delete sample;
    samples->pop_back();
  }
  delete samples;
  delete sampleZs;
  delete invgSampler;
  delete mvGaussianSampler;
  delete data;
  delete testData;
  commit_points.clear();
}

void HybridMedLDA::updateZ(SampleZ* nextZ, CorpusData* dt) {
  double weights[K];                      // weights for importance sampling.
  double A1, A2[K], A3, B1, B2;          // replacements for fast computation.
  int word, N;
  double sel, cul;
  int seli;
  int batchIdx = 0, batchSize = dt->D;
  for(int ii = batchIdx; ii < batchIdx+batchSize; ii++) {
    int i = ii%dt->D;
    N = dt->W[i];
    double invN = 1.0/N, invNx2 = invN*invN;
    for(int k1 = 0; k1 < K; k1++) {
      A2[k1] = 0;
      for(int k2 = 0; k2 < K; k2++) {
        A2[k1] += 2*eta_cov[k1][k2]*nextZ->Cdk[i][k2];
      }
    }
    for(int j = 0; j < N; j++) {
      word = dt->data[i][j];
      if(word >= T) {
        debug("error: word %d out of range [0, %d).\n", word, T);
      }
      nextZ->Cdk[i][nextZ->Z[i][j]]--; // exclude Zij.
      A1 = 0;
      for(int k = 0; k < K; k++) {
        A1 += 2*eta_mean[k]*nextZ->Cdk[i][k];
        A2[k] -= 2*eta_cov[k][nextZ->Z[i][j]];
      }
      B1 = c*dt->y[i]*(1+c*l*nextZ->invlambda[i])*invN;
      B2 = c*c*nextZ->invlambda[i]*0.5*invNx2;
      int flagZ = -1, flag0 = -1; // flag for abnomality.
      for(int k = 0; k < K; k++) {
        A3 = eta_mean[k]*eta_mean[k]+eta_cov[k][k];
        if(k == 0) cul = 0;
        else cul = weights[k-1];
        weights[k] = cul+(nextZ->Cdk[i][k]+alpha0)
              /* strategy 1 variational optimal distribution */
              // *exp(digamma(beta0+gamma[k][word])-digamma(beta0*T+gammasum[k]))
              /* strategy 2 approximation that does not require digamma() */
              *(beta0+gamma[k][word])/(beta0*T+gammasum[k])
              *exp(B1*eta_mean[k]-B2*(A3+(A1*eta_mean[k]+A2[k])));
        if(std::isnan(weights[k])) {
          debug("error: Z weights nan.\n");
          flagZ = k;
        }
        if(std::isinf(weights[k])) flagZ = k; // too discriminative, directly set val.
        if(weights[k] > 0) flag0 = 1;
      }
      if(flagZ >= 0) nextZ->Z[i][j] = flagZ;
      else if(flag0 == -1) nextZ->Z[i][j] = cokus.randomMT()%K;
      else {
        sel = weights[K-1]*cokus.random01();
        for(seli = 0; weights[seli] < sel; seli++);
        nextZ->Z[i][j] = seli;
      }
      for(int k = 0; k < K; k++) {
        A2[k] += 2*eta_cov[k][nextZ->Z[i][j]];
      }
      nextZ->Cdk[i][nextZ->Z[i][j]]++; // restore Cdk, Ckt.
    }
  }
}

void HybridMedLDA::updateLambda(SampleZ *prevZ, CorpusData *dt) {
  int batchIdx = 0, batchSize = dt->D;
  for(int ii = batchIdx; ii < batchIdx+batchSize; ii++) {
    int i = ii%dt->D;
    if(data->W[i] == 0) {
      debug("[error] document length 0\n");
    }
    double discriFunc = 0;
    for(int k = 0; k < K; k++)
      discriFunc += eta_mean[k]*prevZ->Cdk[i][k]/(double)dt->W[i];
    double zetad = l-data->y[i]*discriFunc;
    double bilinear = 0;
    for(int k1 = 0; k1 < K; k1++) {
      for(int k2 = 0; k2 < K; k2++) {
        bilinear += prevZ->Cdk[i][k1]*prevZ->Cdk[i][k2]*eta_cov[k1][k2]/(double)dt->W[i]/(double)dt->W[i];
      }
    }
    invgSampler->reset(1/c/sqrt(zetad*zetad+bilinear), 1);
    prevZ->invlambda[i] = invgSampler->sample();
  }
}

void HybridMedLDA::computeZbar(CorpusData* data, SampleZ *Z, int di) {
  if(Z->Zbar[di] == 0)
    Z->Zbar[di] = new double[K];
  memset(Z->Zbar[di], 0, sizeof(double)*K);
  for(int j = 0; j < data->W[di]; j++) {
    Z->Zbar[di][Z->Z[di][j]]++;
  }
  for(int k = 0; k < K; k++) Z->Zbar[di][k] /= data->W[di]; // normalize.
}

double HybridMedLDA::computeDiscriFunc(CorpusData* dt, int di, Sample *sample, SampleZ *Z, double norm) {
  double discriFunc = 0;
  for(int k = 0; k < K; k++) {
    discriFunc += sample->eta[k]*Z->Cdk[di][k];
  }
  if(norm == 0)
    return discriFunc/(double)dt->W[di];
  else
    return discriFunc/(double)norm;
}

void HybridMedLDA::draw_Z_test(SampleZ* prevZ, int i, CorpusData* dt) {
  // setting basic parameters for convenience.
  int *W = dt->W;
  double sel;
  int seli;
  
  // statistics Cdk.
  double weights[K]; // weights for importance sampling.
  for(int j = 0; j < W[i]; j++) {
    int t = dt->data[i][j];
    if(t >= T) {
      debug("error: word %d out of range [0, %d).\n", t, T);
    }
    prevZ->Cdk[i][prevZ->Z[i][j]]--; // exclude Zij.
    Ckt_test[prevZ->Z[i][j]][t]--;
    Ckt_test_sum[prevZ->Z[i][j]]--;
    int flagZ = -1, flag0 = -1;
    double cum = 0;
    for(int k = 0; k < K; k++) {
      weights[k] = cum+(prevZ->Cdk[i][k]+alpha0)*(beta0+gamma[k][t]+Ckt_test[k][t])/(beta0*T+gammasum[k]+Ckt_test_sum[k]);
      cum = weights[k];
      if(std::isnan(weights[k])) {
        debug("error: Z weights nan.\n");
      }
      if(std::isinf(weights[k])) flagZ = k; // too discriminative, directly set val.
      if(weights[k] > 0) flag0 = 1;
    }
    if(flagZ >= 0) prevZ->Z[i][j] = flagZ;
    else if(flag0 == -1) prevZ->Z[i][j] = cokus.randomMT()%K;
    else {
      sel = weights[K-1]*cokus.random01();
      for(seli = 0; weights[seli] < sel; seli++);
      prevZ->Z[i][j] = seli;
    }
    prevZ->Cdk[i][prevZ->Z[i][j]]++; // restore Cdk, Ckt.
    Ckt_test[prevZ->Z[i][j]][t]++;
    Ckt_test_sum[prevZ->Z[i][j]]++;
  }
  memset(prevZ->Zbar[i], 0, sizeof(double)*K);
  for(int j = 0; j < W[i]; j++) {
    prevZ->Zbar[i][prevZ->Z[i][j]]++;
  }
  for(int k = 0; k < K; k++) prevZ->Zbar[i][k] /= W[i]; // normalize.
}


void HybridMedLDA::infer_Phi_Eta(SampleZ* prevZ, CorpusData* dt, bool reset) {
  int batchIdx = 0, batchSize = dt->D;

  /* setting basic parameters for convenience. */
  if(reset) {
    memset(stat_pmean, 0, sizeof(double)*K);
    for(int k1 = 0; k1 < K; k1++) {
      memset(stat_icov[k1], 0, sizeof(double)*K);
      memset(stat_phi[k1], 0, sizeof(double)*T);
    }
    stat_phi_list_end = -1;
  }

  /* update eta, which is gaussian distribution. */
  for(int k = 0; k < K; k++) {
    for(int dd = batchIdx; dd < batchIdx+batchSize; dd++) {
      int d = dd%dt->D;
      stat_pmean[k] += c*(1+c*l*prevZ->invlambda[d])*dt->y[d]*prevZ->Cdk[d][k]/(double)dt->W[d];
    }
  }
  for(int k1 = 0; k1 < K; k1++) {
    for(int k2 = 0; k2 < K; k2++) {
      for(int dd = batchIdx; dd < batchIdx+batchSize; dd++) {
        int d = dd%dt->D;
        stat_icov[k1][k2] += c*c*prevZ->Cdk[d][k1]*prevZ->Cdk[d][k2]*prevZ->invlambda[d]/(double)dt->W[d]/(double)dt->W[d];
      }
    }
  }

  /* update phi, which is dirichlet distribution. */
  for(int dd = batchIdx; dd < batchIdx+batchSize; dd++) {
    int d = dd%dt->D;
    for(int i = 0; i < dt->W[d]; i++) {
      int k = prevZ->Z[d][i], t = dt->data[d][i];
      if(stat_phi[k][t] == 0) {
        stat_phi_list_end++;
        stat_phi_list_k[stat_phi_list_end] = k;
        stat_phi_list_t[stat_phi_list_end] = t;
      }
      stat_phi[k][t]++;
    }
  }
}

void HybridMedLDA::normalize_Phi_Eta(int N, bool remove) {
  double** eta_lowertriangle = new double*[K];
  for(int i = 0; i < K; i++) {
    eta_lowertriangle[i] = new double[K];
  }
  // %%%%%%%%%%%% normalize eta, which is gaussian distribution.
  for(int k = 0; k < K; k++) {
    if(remove) eta_pmean[k] -= prev_eta_pmean[k];
    prev_eta_pmean[k] = stat_pmean[k]*data->D/(double)batchSize/(double)N;
    eta_pmean[k] += prev_eta_pmean[k];
  }
  for(int k1 = 0; k1 < K; k1++) {
    for(int k2 = 0; k2 < K; k2++) {
      if(remove) eta_icov[k1][k2] -= prev_eta_icov[k1][k2];
      prev_eta_icov[k1][k2] = stat_icov[k1][k2]*data->D/(double)batchSize/(double)N;
      eta_icov[k1][k2] += prev_eta_icov[k1][k2];
    }
  }
  // %%%%%%%%%%%% update phi, which is dirichlet distribution.
  if(remove) {
    for(int stat_i = 0; stat_i <= prev_gamma_list_end; stat_i++) {
      int k = prev_gamma_list_k[stat_i], t = prev_gamma_list_t[stat_i];
      gamma[k][t] -= prev_gamma[k][t];
      gammasum[k] -= prev_gamma[k][t];
    }
  }
  prev_gamma_list_end = -1;
  for(int stat_i = 0; stat_i <= stat_phi_list_end; stat_i++) {
    int k = stat_phi_list_k[stat_i], t = stat_phi_list_t[stat_i];
    prev_gamma[k][t] = stat_phi[k][t]*data->D/(double)batchSize/(double)N;
    prev_gamma_list_end++;
    prev_gamma_list_k[prev_gamma_list_end] = k;
    prev_gamma_list_t[prev_gamma_list_end] = t;
    gamma[k][t] += prev_gamma[k][t];
    gammasum[k] += prev_gamma[k][t];
  }
  // %%%%%%%%%%%% compute aux information.
  inverse_cholydec(eta_icov, eta_cov, eta_lowertriangle, K);
  for(int k = 0; k < K; k++) {
    eta_mean[k] = dotprod(eta_cov[k], eta_pmean, K);
  }
  for(int i = 0; i < K; i++) {
    delete[] eta_lowertriangle[i];
  }
  delete[] eta_lowertriangle;
}
  

double HybridMedLDA::train(vec2D<int> batch, vec<int> label) {
  clock_t time_start = clock();
  clock_t time_end;

  /* create data samples from raw batch */
  int data_size = batch->size();
  DataSample** data_sample = new DataSample*[data_size];
  for(int di = 0; di < data_size; di++) {
    data_sample[di] = new DataSample((*batch)[di], (*label)[di]);
  }
  CorpusData* data = new CorpusData(data_sample, data_size, this->category);
  SampleZ* iZ = new SampleZ(data->D, data->W);
  iZ->Cdk = new double*[data->D];
  for(int i = 0; i < data->D; i++) {
    iZ->Cdk[i] = new double[K];
  }
  for(int d = 0; d < data->D; d++) {
    memset(iZ->Cdk[d], 0, sizeof(double)*K);
    for(int w = 0; w < data->W[d]; w++) {
      iZ->Z[d][w] = cokus.randomMT()%K;
      iZ->Cdk[d][iZ->Z[d][w]]++;
    }
    computeZbar(data, iZ, d);
  }
  updateLambda(iZ, data);

  /* inference via streaming MedLDA */
  for(int si = 0; si < I; si++) {
    for(int sj = 0; sj < J; sj++) {
      updateZ(iZ, data);
      updateLambda(iZ, data);
      if(sj < J_burnin) continue;
      infer_Phi_Eta(iZ, data, sj==J_burnin);
    }
    normalize_Phi_Eta(J-J_burnin, si>0);
  }
  /* cleaning */
  for(int di = 0; di < data_size; di++) {
    delete data_sample[di];
  }
  delete[] data_sample;
  delete data;
  delete iZ;

  time_end = clock();
  train_time += (double)(time_end-time_start)/CLOCKS_PER_SEC;
  return train_time;
}

vector<double> HybridMedLDA::inference(vec2D<int> batch, int num_test_sample, int category) {
  /* pre-clearning */
  Ckt_test.clear();
  Ckt_test_sum.clear();
  Zbar_test.clear();

  /* parse data */
  int data_size = batch->size();
  vector<DataSample*> data_sample(data_size);
  for(int di = 0; di < data_size; di++) {
    data_sample[di] = new DataSample((*batch)[di]);
  }
  CorpusData* testData = new CorpusData(&data_sample[0], data_size, this->category);
  SampleZ* iZ_test = new SampleZ(testData->D, testData->W);

  /* initialization for inference */
  if(num_test_sample != -1) {
    testBurninN = num_test_sample;
    max_gibbs_iter = 2*num_test_sample;
  }

  Ckt_test.resize(K);
  Ckt_test_sum.resize(K, 0);
  for(int k = 0; k < K; k++) {
    Ckt_test[k].resize(T, 0);
  }

  Zbar_test.resize(testData->D);
  iZ_test->Cdk = new double*[testData->D];
  for(int d = 0; d < testData->D; d++) {
    Zbar_test[d].resize(K, 0);
    iZ_test->Cdk[d] = new double[K];
    iZ_test->Zbar[d] = new double[K];
    memset(iZ_test->Cdk[d], 0, sizeof(double)*K);
    memset(iZ_test->Zbar[d], 0, sizeof(double)*K);

    for(int w = 0; w < testData->W[d]; w++) {
      iZ_test->Z[d][w] = cokus.randomMT()%K;
      Ckt_test[iZ_test->Z[d][w]][testData->data[d][w]]++;
      Ckt_test_sum[iZ_test->Z[d][w]]++;
      iZ_test->Cdk[d][iZ_test->Z[d][w]]++;
    }
  }

  /* sample Z with Gibbs sampling.*/
  int zcount = 0;
  for(int it = 0; it < max_gibbs_iter; it++) {
    for(int d = 0; d < testData->D; d++) {
      draw_Z_test(iZ_test, d, testData);
    }
    if(it < testBurninN) continue;
    zcount++;
    for(int d = 0; d < testData->D; d++) {
      for(int k = 0; k < K; k++) Zbar_test[d][k] += iZ_test->Zbar[d][k];
    }
  }
  for(int d = 0; d < testData->D; d++) {
    for(int k = 0; k < K; k++) Zbar_test[d][k] /= (double)zcount;
  }

  /* evaluate inference accuracy.*/
  int acc = 0;
  vector<double> my;
  for(int i = 0; i < testData->D; i++) {
    double discriFunc = 0;
    for(int k = 0; k < K; k++)
      discriFunc += eta_mean[k]*Zbar_test[i][k];
    testData->my[i] = discriFunc;
    if(discriFunc >= 0) testData->py[i] = 1;
    else testData->py[i] = -1;

    my.push_back(testData->my[i]);
  }

  /* cleaning */
  for(int di = 0; di < data_size; di++) 
    delete data_sample[di];
  delete testData;
  return my;
}

double HybridMedLDA::computeCostFunction(SampleZ *z, CorpusData *dt, int batchIdx, int batchSize) {
  double cost = 0;
  
  /* KL[q(\theta) || q_t(\theta)] */
  double* thetavar[batchSize], thetavarsum[batchSize];
  for(int d = 0; d < batchSize; d++) thetavar[d] = new double[K];
  memset(thetavarsum, 0, sizeof(double)*batchSize);
  for(int dd = batchIdx; dd < batchIdx+batchSize; dd++) {
    int d = dd%dt->D;
    double* thetavar_d = thetavar[dd-batchIdx];
    double& thetavarsum_d = thetavarsum[dd-batchIdx];
    for(int k = 0; k < K; k++) {
      thetavar_d[k] = alpha0+z->Cdk[d][k];
      thetavarsum_d += alpha0+z->Cdk[d][k];
    }
    for(int k = 0; k < K; k++) {
      thetavar_d[k] = digamma(thetavar_d[k])-digamma(thetavarsum_d);
      cost += z->Cdk[d][k]*thetavar_d[k];
    }
    cost += gammaln(K*alpha0)-gammaln(K*alpha0+dt->W[d]);
    for(int k = 0; k < K; k++) {
      cost += gammaln(alpha0+z->Cdk[d][k])-gammaln(alpha0);
    }
  }
  for(int d = 0; d < batchSize; d++) delete thetavar[d];
  
  /* KL(q(\Phi) || q_t(\Phi)) */
  for(int k = 0; k < K; k++) {
    double prev_gamma_sum = gammasum[k];
    for(int t = 0; t < T; t++) {
      cost += prev_gamma[k][t]*(digamma(beta0+gamma[k][t])-digamma(beta0*T+gammasum[k]));
      cost += gammaln(beta0+gamma[k][t])-gammaln(beta0+gamma[k][t]-prev_gamma[k][t]);
      prev_gamma_sum -= gamma[k][t];
    }
    cost += gammaln(beta0*T+prev_gamma_sum)-gammaln(beta0*T+gammasum[k]);
  }
  
  /* KL(q(w) || q_t(w)) */
  double* eta_icov0[K], *eta_cov0[K], *eta_lowertriangle[K];
  for(int k = 0; k < K; k++) {
    eta_icov0[k] = new double[K];
    eta_cov0[k] = new double[K];
    eta_lowertriangle[k] = new double[K];
  }
  double eta_pmean0[K], eta_mean0[K];
  for(int k1 = 0; k1 < K; k1++) {
    eta_pmean0[k1] = eta_pmean[k1]-prev_eta_pmean[k1];
    for(int k2 = 0; k2 < K; k2++) {
      eta_icov0[k1][k2] = eta_icov[k1][k2]-prev_eta_icov[k1][k2];
    }
  }
  inverse_cholydec(eta_icov0, eta_cov0, eta_lowertriangle, K);
  for(int k = 0; k < K; k++) {
    eta_mean0[k] = dotprod(eta_cov0[k], eta_pmean0, K);
  }
  for(int k = 0; k < K; k++) {
    for(int j = 0; j < K; j++) {
      cost += eta_icov[k][j]*eta_cov0[j][k]+(eta_mean[k]-eta_mean0[k])*eta_icov[k][j]*(eta_mean[j]-eta_mean0[j]);
    }
  }
  cost -= K;
  for(int k = 0; k < K; k++) {
    delete[] eta_icov0[k];
    delete[] eta_cov0[k];
    delete[] eta_lowertriangle[k];
  }
  /*likelihood p(X_t | Z_t, \phi) */
  for(int k = 0; k < K; k++) {
    for(int t = 0; t < T; t++) {
      cost -= prev_gamma[k][t]*(digamma(gamma[k][t])-digamma(gammasum[k]));
    }
  }
  
  /*pseudo-likelihood psi(Y_t, \lambdav_t | Z_t, w) */
  for(int dd = batchIdx; dd < batchIdx+batchSize; dd++) {
    int d = dd%dt->D;
    double discri = 0;
    for(int k = 0; k < K; k++) {
      discri += eta_mean[k]*z->Cdk[d][k]/(double)dt->W[d];
    }
    cost += 0-1/2*log(z->invlambda[d]/2/M_PI)+1/2*z->invlambda[d]
                  *(1/z->invlambda[d]+c*discri)
                  *(1/z->invlambda[d]+c*discri);
  }
  return cost;
}

//////////////////////////////////////////////////////////////////////
///////////// private methods ////////////////////////////////////////
