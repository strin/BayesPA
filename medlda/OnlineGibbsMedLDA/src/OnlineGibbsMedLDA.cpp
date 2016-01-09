
//  GibbsSampler.cpp
//  OnlineTopic
//
//  Created by Tianlin Shi on 5/1/13.
//  Copyright (c) 2013 Tianlin Shi. All rights reserved.
//

#include "OnlineGibbsMedLDA.h"
#include "utils.h"
#include "gammaln.h"

using namespace stl;
using namespace std;

OnlineGibbsMedLDA::OnlineGibbsMedLDA(const vector<int>& category) {
  invgSampler = new InverseGaussian();
  mvGaussianSampler = new MVGaussian();
  
  /* parameters by default */
  K                  = 5;	    
  T                  = 0;
  this->category = category; // the categories to learn classifier for.
  num_category       = 2; // can be different than category.size()

  I                  = 3;           
  J                  = 1;	    
  J_burnin           = 0;          
  l                  = 164;
  alpha0             = 0.5;      
  beta0              = 0.45;      
  v                  = 1;
  c                  = 1;
  stepsize           = 1;
  point_estimate_for_test = false;
}

void OnlineGibbsMedLDA::init() {
  /* init global variables and sufficient stats */
  gamma = vec2D<double>(K);
  prev_gamma = vec2D<double>(K); 
  gammasum = vec<double>(K, 0);
  for(int k = 0; k < K; k++) {
    gamma[k].resize(T, 0);
    prev_gamma[k].resize(T, 0);
  }

  stat_phi = vec2D<double>(K);
  for(int i = 0; i < K; i++) 
    stat_phi[i].resize(T, 0);

  prev_gamma_list_k = vec<int>();
  prev_gamma_list_t = vec<int>();
  stat_phi_list_k = vec<int>();
  stat_phi_list_t = vec<int>();

  this->weight.resize(category.size());
  for(size_t ci = 0; ci < category.size(); ci++) {
    this->weight[ci] = make_shared<SampleWeight>(K, v);
  }

  train_time = 0;

  /* init random source */
  cokus.reloadMT();
  cokus.seedMT(time(NULL));
  invgSampler->reset(1, 1);
}

OnlineGibbsMedLDA::~OnlineGibbsMedLDA() {
  /* clean stat */
  delete invgSampler;
  delete mvGaussianSampler;
}

void OnlineGibbsMedLDA::updateZ(ptr<SampleZ> nextZ, ptr<CorpusData> dt) {
  double weights[K], log_weights[K];          // weights for importance sampling.
  double A1[category.size()], A2[category.size()][K], 
         A3[category.size()], B1[category.size()], B2[category.size()];      // replacements for fast computation.
  int word, N;
  double sel;
  int seli;
  int batchIdx = 0, batchSize = dt->D;
  for(int ii = batchIdx; ii < batchIdx+batchSize; ii++) {
    int i = ii%dt->D;
    N = dt->W[i];
    double invN = 1.0/N, invNx2 = invN*invN;

    for(size_t ci = 0; ci < category.size(); ci++) {
      for(int k1 = 0; k1 < K; k1++) {
        A2[ci][k1] = 0;
        for(int k2 = 0; k2 < K; k2++) {
          A2[ci][k1] += 2*weight[ci]->eta_cov[k1][k2]*nextZ->Cdk[i][k2];
        }
      }
    }

    for(int j = 0; j < N; j++) {
      word = dt->data[i][j];
      if(word >= T) {
        debug("error: word %d out of range [0, %d).\n", word, T);
      }
      nextZ->Cdk[i][nextZ->Z[i][j]]--; // exclude Zij.

      for(size_t ci = 0; ci < category.size(); ci++) {
        A1[ci] = 0;
        for(int k = 0; k < K; k++) {
          A1[ci] += 2*weight[ci]->eta_mean[k]*nextZ->Cdk[i][k];
          A2[ci][k] -= 2*weight[ci]->eta_cov[k][nextZ->Z[i][j]];
        }
        B1[ci] = c*dt->y[i][category[ci]]*(1+c*l*nextZ->invlambda[ci][i])*invN;
        B2[ci] = c*c*nextZ->invlambda[ci][i]*0.5*invNx2;
      }

      double max_log_weight =  -1e10;

      for(int k = 0; k < K; k++) {
        for(size_t ci = 0; ci < category.size(); ci++) {
          A3[ci] = weight[ci]->eta_mean[k]*weight[ci]->eta_mean[k]+weight[ci]->eta_cov[k][k];
        }

        log_weights[k] = log((nextZ->Cdk[i][k]+alpha0) 
                /* strategy 1 variational optimal distribution */
                // *exp(digamma(beta0+gamma[k][word])-digamma(beta0*T+gammasum[k]))
                /* strategy 2 approximation that does not require digamma() */
                *(beta0+gamma[k][word])/(beta0*T+gammasum[k])
            );

        for(size_t ci = 0; ci < category.size(); ci++) {
          log_weights[k] += B1[ci] * weight[ci]->eta_mean[k] 
            - B2[ci] * (A3[ci] + (A1[ci] * weight[ci]->eta_mean[k]
                  + A2[ci][k]));
        }

        if(log_weights[k] > max_log_weight) max_log_weight = log_weights[k];
      }

      double cum = 0;
      for(int k = 0; k < K; k++) {
        log_weights[k] -= max_log_weight;
        weights[k] = cum + exp(log_weights[k]);
        cum = weights[k];
      }
      
      sel = cum * cokus.random01();
      for(seli = 0; weights[seli] < sel; seli++);
      nextZ->Z[i][j] = seli;
    
      for(size_t ci = 0; ci < category.size(); ci++) {
        for(int k = 0; k < K; k++) {
          A2[ci][k] += 2 * weight[ci]->eta_cov[k][nextZ->Z[i][j]];
        }
      }

      nextZ->Cdk[i][nextZ->Z[i][j]]++; // restore Cdk, Ckt.
    }
  }
}

void OnlineGibbsMedLDA::updateLambda(ptr<SampleZ> prevZ, ptr<CorpusData> dt) {
  int batchIdx = 0, batchSize = dt->D;
  for(size_t ci = 0; ci < category.size(); ci++) {
    for(int ii = batchIdx; ii < batchIdx+batchSize; ii++) {
      int i = ii%dt->D;
      double discriFunc = 0;
      for(int k = 0; k < K; k++)
        discriFunc += weight[ci]->eta_mean[k]*prevZ->Cdk[i][k]/(double)dt->W[i];
      double zetad = l-dt->y[i][category[ci]]*discriFunc;
      double bilinear = 0;
      for(int k1 = 0; k1 < K; k1++) {
        for(int k2 = 0; k2 < K; k2++) {
          bilinear += prevZ->Cdk[i][k1]*prevZ->Cdk[i][k2]*weight[ci]->eta_cov[k1][k2]/(double)dt->W[i]/(double)dt->W[i];
        }
      }
      invgSampler->reset(1/c/sqrt(zetad*zetad+bilinear), 1);
      prevZ->invlambda[ci][i] = invgSampler->sample();
    }
  }
}

void OnlineGibbsMedLDA::computeZbar(ptr<CorpusData> data, ptr<SampleZ> Z, int di) {
  if(Z->Zbar[di] == 0)
    Z->Zbar[di] = new double[K];
  memset(Z->Zbar[di], 0, sizeof(double)*K);
  for(int j = 0; j < data->W[di]; j++) {
    Z->Zbar[di][Z->Z[di][j]]++;
  }
  for(int k = 0; k < K; k++) Z->Zbar[di][k] /= data->W[di]; // normalize.
}

void OnlineGibbsMedLDA::draw_Z_test(ptr<SampleZ> prevZ, int i, ptr<CorpusData> dt, vec2D<double>& Ckt, vec<double>& Ckt_sum) {
  // setting basic parameters for convenience.
  auto W = dt->W;
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
    // update test topic stats matrix.
    Ckt[prevZ->Z[i][j]][t]--;
    Ckt_sum[prevZ->Z[i][j]]--;
    int flagZ = -1, flag0 = -1;
    double cum = 0;
    for(int k = 0; k < K; k++) {
      weights[k] = cum+(prevZ->Cdk[i][k]+alpha0)
          * (beta0 + gamma[k][t] + double(!point_estimate_for_test) * Ckt[k][t])
          / (beta0 * T + gammasum[k] + double(!point_estimate_for_test) * Ckt_sum[k]);
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
    Ckt[prevZ->Z[i][j]][t]++;
    Ckt_sum[prevZ->Z[i][j]]++;
  }
  memset(prevZ->Zbar[i], 0, sizeof(double)*K);
  for(int j = 0; j < W[i]; j++) {
    prevZ->Zbar[i][prevZ->Z[i][j]]++;
  }
  for(int k = 0; k < K; k++) prevZ->Zbar[i][k] /= W[i]; // normalize.
}


void OnlineGibbsMedLDA::infer_Phi_Eta(ptr<SampleZ> prevZ, ptr<CorpusData> dt, bool reset) {
  int batchIdx = 0, batchSize = dt->D;

  /* setting basic parameters for convenience. */
  if(reset) {
    for(size_t ci = 0; ci < category.size(); ci++) {
      memset(&weight[ci]->stat_pmean[0], 0, sizeof(double)*K);
      for(int k1 = 0; k1 < K; k1++) {
        memset(&weight[ci]->stat_icov[k1][0], 0, sizeof(double)*K);
      }
    }

    for(int k1 = 0; k1 < K; k1++) {
      memset(&stat_phi[k1][0], 0, sizeof(double)*T);
    }
    stat_phi_list_k.clear();
    stat_phi_list_t.clear();
  }

  /* update eta, which is gaussian distribution. */
  for(size_t ci = 0; ci < category.size(); ci++) {
    for(int k = 0; k < K; k++) {
      for(int dd = batchIdx; dd < batchIdx+batchSize; dd++) {
        int d = dd%dt->D;
        weight[ci]->stat_pmean[k] += c*(1+c*l*prevZ->invlambda[ci][d])*dt->y[d][category[ci]]*prevZ->Cdk[d][k]/(double)dt->W[d];
      }
    }
    for(int k1 = 0; k1 < K; k1++) {
      for(int k2 = 0; k2 < K; k2++) {
        for(int dd = batchIdx; dd < batchIdx+batchSize; dd++) {
          int d = dd%dt->D;
          weight[ci]->stat_icov[k1][k2] += c*c*prevZ->Cdk[d][k1]*prevZ->Cdk[d][k2]*prevZ->invlambda[ci][d]/(double)dt->W[d]/(double)dt->W[d];
        }
      }
    }
  }

  /* update phi, which is dirichlet distribution. */
  for(int dd = batchIdx; dd < batchIdx+batchSize; dd++) {
    int d = dd%dt->D;
    for(int i = 0; i < dt->W[d]; i++) {
      int k = prevZ->Z[d][i], t = dt->data[d][i];
      if(stat_phi[k][t] == 0) {
        stat_phi_list_k.push_back(k);
        stat_phi_list_t.push_back(t);
      }
      stat_phi[k][t]++;
    }
  }
}

void OnlineGibbsMedLDA::normalize_Phi_Eta(int N, bool remove) {
  /* normalize eta, which is gaussian distribution. */
  for(size_t ci = 0; ci < category.size(); ci++) {
    for(int k = 0; k < K; k++) {
      if(remove) weight[ci]->eta_pmean[k] -= weight[ci]->prev_eta_pmean[k];
      weight[ci]->prev_eta_pmean[k] = weight[ci]->stat_pmean[k] * stepsize / (double)N;
      weight[ci]->eta_pmean[k] += weight[ci]->prev_eta_pmean[k];
    }

    for(int k1 = 0; k1 < K; k1++) {
      for(int k2 = 0; k2 < K; k2++) {
        if(remove) weight[ci]->eta_icov[k1][k2] -= weight[ci]->prev_eta_icov[k1][k2];
        weight[ci]->prev_eta_icov[k1][k2] = weight[ci]->stat_icov[k1][k2] * stepsize / (double)N;
        weight[ci]->eta_icov[k1][k2] += weight[ci]->prev_eta_icov[k1][k2];
      }
    }
  }

  /* update phi, which is dirichlet distribution. */
  if(remove) {
    for(size_t stat_i = 0; stat_i < prev_gamma_list_k.size(); stat_i++) {
      int k = prev_gamma_list_k[stat_i], t = prev_gamma_list_t[stat_i];
      gamma[k][t] -= prev_gamma[k][t];
      gammasum[k] -= prev_gamma[k][t];
    }
  }

  prev_gamma_list_k.clear();
  prev_gamma_list_t.clear();

  for(size_t stat_i = 0; stat_i < stat_phi_list_k.size(); stat_i++) {
    int k = stat_phi_list_k[stat_i], t = stat_phi_list_t[stat_i];

    prev_gamma[k][t] = stat_phi[k][t] * stepsize / (double)N;

    prev_gamma_list_k.push_back(k);
    prev_gamma_list_t.push_back(t);

    gamma[k][t] += prev_gamma[k][t];
    gammasum[k] += prev_gamma[k][t];
  }

  /* compute aux information. */
  for(size_t ci = 0; ci < category.size(); ci++) {
    vec2D<double> eta_lowertriangle(K);
    for(int i = 0; i < K; i++) 
      eta_lowertriangle[i].resize(K, 0);

    inverse_cholydec(weight[ci]->eta_icov, weight[ci]->eta_cov, eta_lowertriangle, K);
    for(int k = 0; k < K; k++) {
      weight[ci]->eta_mean[k] = dotprod(weight[ci]->eta_cov[k], *weight[ci]->eta_pmean, K);
    }
  }
}
  

double OnlineGibbsMedLDA::train(const vec2D<int>& batch, const vec<int>& labels) {
  vec2D<int> multi_labels;
  for(auto label: labels) {
    vector<int> this_label;
    this_label.push_back(label);
    multi_labels.push_back(this_label);
  }
  return this->train(batch, multi_labels);
}


double OnlineGibbsMedLDA::train(const vec2D<int>& batch, const vec2D<int>& labels) {
  clock_t time_start = clock();
  clock_t time_end;

  /* create data samples from raw batch */
  ptr<CorpusData> data = make_shared<CorpusData>(batch, labels, this->num_category);

  /* initialize the samples of latent variables */
  ptr<SampleZ> iZ = make_shared<SampleZ>(data->D, data->W, category.size());
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

  /* training via streaming MedLDA */
  for(int si = 0; si < I; si++) {
    for(int sj = 0; sj < J; sj++) {
      updateZ(iZ, data);
      updateLambda(iZ, data);
      if(sj < J_burnin) continue;
      infer_Phi_Eta(iZ, data, sj==J_burnin);
    }
    normalize_Phi_Eta(J-J_burnin, si>0);
  }

  time_end = clock();
  train_time += (double)(time_end-time_start)/CLOCKS_PER_SEC;
  return train_time;
}

vec2D<double> OnlineGibbsMedLDA::inference(vec2D<int> batch, int num_test_sample) {
  /* pre-clearning */
  Zbar_test.clear();

  /* parse data */
  ptr<CorpusData> testData = make_shared<CorpusData>(batch, this->num_category);
  size_t data_size = testData->D;
  ptr<SampleZ> iZ_test = make_shared<SampleZ>(testData->D, testData->W, category.size());

  /* initialization for inference */
  int testBurninN, max_gibbs_iter;
  if(num_test_sample != -1) {
    testBurninN = num_test_sample;
    max_gibbs_iter = 2*num_test_sample;
  }

  // initialize Ckt for test.
  vec2D<double> Ckt_test = vec2D<double>(K);
  for( int k = 0; k < K; k++) {
    Ckt_test[k].resize(T, 0);
  }
  vec<double> Ckt_test_sum = vec<double>(K, 0);

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
      iZ_test->Cdk[d][iZ_test->Z[d][w]]++;
      Ckt_test[iZ_test->Z[d][w]][testData->data[d][w]]++;
      Ckt_test_sum[iZ_test->Z[d][w]]++;
    }
  }

  /* sample Z with Gibbs sampling.*/
  int zcount = 0;
  for(int it = 0; it < max_gibbs_iter; it++) {
    for(int d = 0; d < testData->D; d++) {
      draw_Z_test(iZ_test, d, testData, Ckt_test, Ckt_test_sum);
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
  vec2D<double> my;

  for(int i = 0; i < testData->D; i++) {
    vector<double> resp;

    for(size_t ci = 0; ci < category.size(); ci++) {
      double discriFunc = 0;
      for(int k = 0; k < K; k++)
        discriFunc += weight[ci]->eta_mean[k]*Zbar_test[i][k];
      testData->my[i][category[ci]] = discriFunc;
      if(discriFunc >= 0) testData->py[i][category[ci]] = 1;
      else testData->py[i][category[ci]] = -1;

      resp.push_back(discriFunc);
    }
  
    my.push_back(resp);
  }

  return my;
}


//////////////////////////////////////////////////////////////////////
///////////// private methods ////////////////////////////////////////
