#include "paMedLDAgibbs.h"
#include "utils.h"
#include "gammaln.h"

#define GEN_BIN_LABEL(y,m_category) (2*((int)(y) == (m_category))-1)
#define SIGN(val) ((double)((val) > 0)-(double)((val) < 0))

paMedLDAgibbs::paMedLDAgibbs(Corpus* corpus, int category) {
	this->corpus = corpus;
	this->category = category;
	samples = new deque<Sample*>();
	sampleZs = new deque<SampleZ*>();
	invgSampler = new InverseGaussian();
	mvGaussianSampler = new MVGaussian();
	
	train_data = &corpus->train_data;
	test_data = &corpus->test_data;
	
	// set basic parameters.
	T									= (int)corpus->m_T;
	
	// key parameters.
	K									= 5; // number of topics.
	batchSize							= train_data->D;   // mini-batch size.
	epoch								= 1;     // number of scans of the entire corpus.
	I									= 1;
	J									= 3;   // Z repeated sample size.
	u									= 1;
	tao									= 1;
	J_burnin							= 0;
	l									= 164;
	max_gibbs_iter						= 50;
	testBurninN							= 10;
	alpha0								= 1/(double)K;  // prior of document topic distribution.
	beta0								= 0.5;			// prior of dictionary.

	// other parameters.
	v									= 1;
	c									= 1;
	lets_commit							= false;
	lets_batch							= false;
	lets_multic							= false;
	mode_additive						= false;
	commit_point_spacing				= 1;
	
	cokus.reloadMT();
	cokus.seedMT(time(NULL)+category);
}

void paMedLDAgibbs::init() {
	/* sampling setting */
	maxSampleN							= 1;
	if(lets_commit) {
		maxBurninN						= commit_point_spacing*commit_point_n;
	}else{
		maxBurninN						= epoch*train_data->D/batchSize+1;
	}
		
	/* train_data init*/
	my = new double[test_data->D];
	py = new double[test_data->D];
	
	
	/* stat init*/
	Cdk_test = new double*[test_data->D];				
	for( int i = 0; i < test_data->D; i++) {
		Cdk_test[i] = new double[K];
		memset( Cdk_test[i], 0, sizeof(double)*K);
	}
	eta_icov = new double*[K];
	eta_cov = new double*[K];
	prev_eta_icov = new double*[K];
	eta_pmean = new double[K];
	eta_mean = new double[K];
	stat_pmean = new double[K];
	prev_eta_pmean = new double[K];
	memset(stat_pmean, 0, sizeof(double)*K);
	memset( eta_pmean, 0, sizeof(double)*K);
	memset( eta_mean, 0, sizeof(double)*K);
	stat_icov = new double*[K];
	for( int k = 0; k < K; k++) {
		prev_eta_icov[k] = new double[K];
		eta_cov[k] = new double[K];
		eta_icov[k] = new double[K];
		memset( eta_icov[k], 0, sizeof(double)*K);
		eta_icov[k][k] = 1/(v*v);
		stat_icov[k] = new double[K];
		memset(stat_icov[k], 0, sizeof(double)*K);
	}
	gamma = new double*[K];
	prev_gamma = new double*[K];
	gammasum = new double[K];
	for( int k = 0; k < K; k++) {
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
	for( int i = 0; i < K; i++) {
		stat_phi[i] = new double[T];
		memset(stat_phi[i], 0, sizeof(double)*T);
	}
	Zbar_test = new double*[test_data->D];
	for( int d = 0; d < test_data->D; d++) {
		Zbar_test[d] = new double[K];
		memset( Zbar_test[d], 0, sizeof(double)*K);
	}
	iZ_test = new SampleZ(test_data);
	forget_factor = new double[maxBurninN+maxSampleN];

	commit_points.clear();

	invgSampler->reset(1, 1);
	iZ = new SampleZ(train_data);
	iZ->Cdk = new double*[train_data->D];
	for( int i = 0; i < train_data->D; i++) {
		iZ->Cdk[i] = new double[K];
	}
	for( int d = 0; d < train_data->D; d++) {
		memset(iZ->Cdk[d], 0, sizeof(double)*K);
		for( int w = 0; w < train_data->doc[d].nd; w++) {
			iZ->Z[d][w] = cokus.randomMT()%K;
			iZ->Cdk[d][iZ->Z[d][w]]++;
		}
		computeZbar(train_data, iZ, d);
	}
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
	for( int k = 0; k < K; k++) {
		gammasum[k] = 0;
		memset(stat_phi[k], 0, sizeof(double)*T);
		memset(gamma[k], 0, sizeof(double)*T);
	}
	updateLambda(iZ, 0, train_data->D);
	
	batchIdx = 0;
	train_time = 0;
}

paMedLDAgibbs::~paMedLDAgibbs() {
	for( int i = 0; i < test_data->D; i++) {
		delete[] Cdk_test[i];
		delete[] Zbar_test[i];
	}
	delete[] Zbar_test;
	delete[] Cdk_test;
	
	/* clean stat */
	for( int i = 0; i < K; i++) {
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
	for( int i = 0; i < samples->size(); i++) {
		Sample* sample = samples->back();
		delete sample;
		samples->pop_back();
	}
	delete[] forget_factor;
	delete iZ_test;
	delete samples;
	delete sampleZs;
	delete invgSampler;
	delete mvGaussianSampler;
	commit_points.clear();
	
	delete[] py;
	delete[] my;
	if(iZ) delete iZ;
}

void paMedLDAgibbs::updateZ(SampleZ* nextZ, int batchIdx, int batchSize) {
	double weights[K];											// weights for importance sampling.
	double A1, A2[K], A3, B1, B2;					// replacements for fast computation.
	double sel, cul;
	int seli;
	for(int ii = batchIdx; ii < batchIdx+batchSize; ii++) {
		int i = ii%train_data->D;
		Document& doc = train_data->doc[i];
		double nd = doc.nd;
		A1 = 0;
		for(int k1 = 0; k1 < K; k1++) {
			A2[k1] = 0;
			for(int k2 = 0; k2 < K; k2++) {
				A2[k1] += 2*eta_cov[k1][k2]*nextZ->Cdk[i][k2];
			}
			A1 += 2*eta_mean[k1]*nextZ->Cdk[i][k1];
		}
		B1 = c*GEN_BIN_LABEL(doc.y[0], category)*(1+c*l*nextZ->invlambda[i])/nd;
		B2 = c*c*nextZ->invlambda[i]*0.5/nd/nd;
		for( int j = 0; j < nd; j++) {
			int word = doc.words[j];
			nextZ->Cdk[i][nextZ->Z[i][j]]--; 
			for(int k = 0; k < K; k++) 
				A2[k] -= 2*eta_cov[k][nextZ->Z[i][j]];
			int flagZ = -1, flag0 = -1; 
			double cum = 0;
			for( int k = 0; k < K; k++) {
				A3 = eta_mean[k]*eta_mean[k]+eta_cov[k][k];
				weights[k] = cum+(nextZ->Cdk[i][k]+alpha0)
							/* strategy 1 variational optimal distribution */
							// *exp(digamma(beta0+gamma[k][word])-digamma(beta0*T+gammasum[k]))
							/* strategy 2 approximation that does not require digamma() */
							*(beta0+gamma[k][word])/(beta0*T+gammasum[k])
							*exp(B1*eta_mean[k]-B2*(A3+(A1*eta_mean[k]+A2[k])));
				cum = weights[k];
			}
			sel = weights[K-1]*cokus.random01();
			for( seli = 0; weights[seli] < sel; seli++);
			nextZ->Z[i][j] = seli;
			for(int k = 0; k < K; k++) 
				A2[k] += 2*eta_cov[k][nextZ->Z[i][j]];
			nextZ->Cdk[i][nextZ->Z[i][j]]++; 
		}
	}
}

void paMedLDAgibbs::updateLambda(SampleZ *prevZ, int batchIdx, int batchSize) {
	for(int ii = batchIdx; ii < batchIdx+batchSize; ii++) {
		int i = ii%train_data->D;
		Document& doc = train_data->doc[i];
		if( doc.nd == 0) 
			debug( "[error] document length 0\n");
		double discriFunc = 0;
		for( int k = 0; k < K; k++)
			discriFunc += eta_mean[k]*prevZ->Cdk[i][k]/(double)doc.nd;
		double zetad = l-GEN_BIN_LABEL(doc.y[0], category)*discriFunc;
		double bilinear = 0;
		for( int k1 = 0; k1 < K; k1++) {
			for( int k2 = 0; k2 < K; k2++) {
				bilinear += prevZ->Cdk[i][k1]*prevZ->Cdk[i][k2]*eta_cov[k1][k2]/(double)doc.nd/(double)doc.nd;
			}
		}
		invgSampler->reset(1/c/sqrt(zetad*zetad+bilinear), 1);
		prevZ->invlambda[i] = invgSampler->sample();
	}
}

void paMedLDAgibbs::computeZbar(CorpusData* train_data, SampleZ *Z, int di) {
	if( Z->Zbar[di] == 0)
		Z->Zbar[di] = new double[K];
	memset( Z->Zbar[di], 0, sizeof(double)*K);
	for( int j = 0; j < train_data->doc[di].nd; j++) {
		Z->Zbar[di][Z->Z[di][j]]++;
	}
	for( int k = 0; k < K; k++) Z->Zbar[di][k] /= (double)train_data->doc[di].nd; // normalize.
}

double paMedLDAgibbs::computeDiscriFunc(CorpusData* dt, int di, Sample *sample, SampleZ *Z, double norm) {
	double discriFunc = 0;
	for( int k = 0; k < K; k++) {
		discriFunc += sample->eta[k]*Z->Cdk[di][k];
	}
	if( norm == 0)
		return discriFunc/(double)dt->doc[di].nd;
	else
		return discriFunc/(double)norm;
}

void paMedLDAgibbs::draw_Z_test(Sample* sample, SampleZ* prevZ, int i, CorpusData* dt) {
	// setting basic parameters for convenience.
	Document& doc = dt->doc[i];
	double sel;
	int seli;
	
	// statistics Cdk.
	double weights[K]; // weights for importance sampling.
	for( int j = 0; j < doc.nd; j++) {
		int t = doc.words[j];
		Cdk_test[i][prevZ->Z[i][j]]--; // exclude Zij.
		Ckt_test[prevZ->Z[i][j]][t]--;
		Ckt_test_sum[prevZ->Z[i][j]]--;
		int flagZ = -1, flag0 = -1;
		double cum = 0;
		for( int k = 0; k < K; k++) {
			weights[k] = cum+(Cdk_test[i][k]+alpha0)*(beta0+gamma[k][t])/(beta0*T+gammasum[k]);
			cum = weights[k];
			if( std::isnan((double)weights[k])) {
				debug( "error: Z weights nan.\n");
			}
			if( std::isinf(weights[k])) flagZ = k; // too discriminative, directly set val.
			if( weights[k] > 0) flag0 = 1;
		}
		if( flagZ >= 0) prevZ->Z[i][j] = flagZ;
		else if( flag0 == -1) prevZ->Z[i][j] = cokus.randomMT()%K;
		else {
			sel = weights[K-1]*cokus.random01();
			for( seli = 0; weights[seli] < sel; seli++);
			prevZ->Z[i][j] = seli;
		}
		Cdk_test[i][prevZ->Z[i][j]]++; // restore Cdk, Ckt.
		Ckt_test[prevZ->Z[i][j]][t]++;
		Ckt_test_sum[prevZ->Z[i][j]]++;
	}
	if( prevZ->Zbar == 0)
		prevZ->Zbar = new double*[test_data->D];
	if( prevZ->Zbar[i] == 0)
		prevZ->Zbar[i] = new double[K];
	memset( prevZ->Zbar[i], 0, sizeof(double)*K);
	for( int j = 0; j < doc.nd; j++) {
		prevZ->Zbar[i][prevZ->Z[i][j]]++;
	}
	for( int k = 0; k < K; k++) prevZ->Zbar[i][k] /= (double)doc.nd; // normalize.
}


void paMedLDAgibbs::infer_Phi_Eta(SampleZ* prevZ, int batchIdx, int batchSize, bool reset) {
	// %%%%%%%%%%%% setting basic parameters for convenience.
	if(reset) {
		memset(stat_pmean, 0, sizeof(double)*K);
		for(int k1 = 0; k1 < K; k1++) {
			memset(stat_icov[k1], 0, sizeof(double)*K);
		}
		stat_phi_list_end = -1;
	}
	// %%%%%%%%%%%% update eta, which is gaussian distribution.
	for( int k = 0; k < K; k++) {
		for(int dd = batchIdx; dd < batchIdx+batchSize; dd++) {
			int d = dd%train_data->D;
			Document& doc = train_data->doc[d];
			stat_pmean[k] += c*(1+c*l*prevZ->invlambda[d])*GEN_BIN_LABEL(doc.y[0], category)*prevZ->Cdk[d][k]/(double)doc.nd;
		}
	}
	for( int k1 = 0; k1 < K; k1++) {
		for( int k2 = 0; k2 < K; k2++) {
			for(int dd = batchIdx; dd < batchIdx+batchSize; dd++) {
				int d = dd%train_data->D;
				Document& doc = train_data->doc[d];
				stat_icov[k1][k2] += c*c*prevZ->Cdk[d][k1]*prevZ->Cdk[d][k2]*prevZ->invlambda[d]/(double)doc.nd/(double)doc.nd;
			}
		}
	}
	// %%%%%%%%%%%% update phi, which is dirichlet distribution.
	for(int dd = batchIdx; dd < batchIdx+batchSize; dd++) {
		int d = dd%train_data->D;
		Document& doc = train_data->doc[d];
		for(int i = 0; i < doc.nd; i++) {
			int k = prevZ->Z[d][i], t = doc.words[i];
			if(stat_phi[k][t] == 0) {
				stat_phi_list_end++;
				stat_phi_list_k[stat_phi_list_end] = k;
				stat_phi_list_t[stat_phi_list_end] = t;
			}
			stat_phi[k][t]++;
		}
	}
}

void paMedLDAgibbs::normalize_Phi_Eta(int N, bool remove) {
	double** eta_lowertriangle = new double*[K];
	for( int i = 0; i < K; i++) {
		eta_lowertriangle[i] = new double[K];
	}
	// %%%%%%%%%%%% normalize eta, which is gaussian distribution.
	for( int k = 0; k < K; k++) {
		if(remove) eta_pmean[k] -= prev_eta_pmean[k];
		prev_eta_pmean[k] = stat_pmean[k]/(double)N;
		eta_pmean[k] += prev_eta_pmean[k];
	}
	for( int k1 = 0; k1 < K; k1++) {
		for( int k2 = 0; k2 < K; k2++) {
			if(remove) eta_icov[k1][k2] -= prev_eta_icov[k1][k2];
			prev_eta_icov[k1][k2] = stat_icov[k1][k2]/(double)N;
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
		prev_gamma[k][t] = stat_phi[k][t]/(double)N;
		prev_gamma_list_end++;
		prev_gamma_list_k[prev_gamma_list_end] = k;
		prev_gamma_list_t[prev_gamma_list_end] = t;
		gamma[k][t] += prev_gamma[k][t];
		gammasum[k] += prev_gamma[k][t];
		stat_phi[k][t] = 0;
	}
	// %%%%%%%%%%%% compute aux information.
	inverse_cholydec(eta_icov, eta_cov, eta_lowertriangle, K);
	for( int k = 0; k < K; k++) {
		eta_mean[k] = dotprod( eta_cov[k], eta_pmean, K);
	}
	for( int i = 0; i < K; i++) {
		delete[] eta_lowertriangle[i];
	}
	delete[] eta_lowertriangle;
}
	

double paMedLDAgibbs::train(int num_iter) {
	clock_t time_start = clock();
	clock_t time_end, ta, tb;
	
	for(int burnin = 0; burnin < num_iter; burnin++, batchIdx += batchSize) {
		for(int si = 0; si < I; si++) {
			for( int sj = 0; sj < J; sj++) {
				/* sample latent assignments */
				updateZ(iZ, batchIdx, batchSize);
				/* sample augmented variables */
				updateLambda(iZ, batchIdx, batchSize);
				/* disgard burnin samples */
				if(sj < J_burnin) continue;
				/* cumulate stats */
				infer_Phi_Eta(iZ, batchIdx, batchSize, sj==J_burnin);
			}
			/* update global with cumulated stats */
			normalize_Phi_Eta(J-J_burnin, si>0);
		}
	}
	// %%%%%%%%%%%% clean
	time_end = clock();
	train_time += (double)(time_end-time_start)/CLOCKS_PER_SEC;
	return train_time;
}

double paMedLDAgibbs::inference(CorpusData* test_data, int num_test_sample) {
	// use one sample of phi and eta.
	// initialize samples of Z randomly.
	testBurninN = num_test_sample;
	max_gibbs_iter = 3*num_test_sample;
    Sample* sample = new Sample(K, T);
	double** eta_lowertriangle = new double*[K];
	Ckt_test = new double*[K];
	for( int i = 0; i < K; i++) {
		eta_lowertriangle[i] = new double[K];
	}
	for( int k = 0; k < K; k++) {
		Ckt_test[k] = new double[T];
		memset(Ckt_test[k], 0, sizeof(double)*T);
//		for(int t = 0; t < T; t++) {
//			sample->phi[k][t] = beta0+gamma[k][t];
//		}
//		for( int t = 0; t < T; t++) sample->phi[k][t] /= (beta0*T+gammasum[k]);  // normalize.
	}
	/* sample Z with Gibbs sampling.*/
	Ckt_test_sum = new double[K];
	memset(Ckt_test_sum, 0, sizeof(double)*K);
	for( int d = 0; d < test_data->D; d++) {
		memset( Zbar_test[d], 0, sizeof(double)*K);
		memset( Cdk_test[d], 0, sizeof(double)*K);
		for( int w = 0; w < test_data->doc[d].nd; w++) {
			iZ_test->Z[d][w] = cokus.randomMT()%K;
			Ckt_test[iZ_test->Z[d][w]][test_data->doc[d].words[w]]++;
			Ckt_test_sum[iZ_test->Z[d][w]]++;
			Cdk_test[d][iZ_test->Z[d][w]]++;
		}
	}
	int zcount = 0;
	for( int it = 0; it < max_gibbs_iter; it++) {
		for( int d = 0; d < test_data->D; d++) {
			draw_Z_test(sample, iZ_test, d, test_data);
		}
		if(it < testBurninN) continue;
		zcount++;
		for(int d = 0; d < test_data->D; d++) {
			for( int k = 0; k < K; k++) Zbar_test[d][k] += iZ_test->Zbar[d][k];
		}
	}
	for(int d = 0; d < test_data->D; d++) {
		for( int k = 0; k < K; k++) Zbar_test[d][k] /= (double)zcount;
	}
	/* evaluate inference accuracy.*/
    double acc = 0;
	for( int i = 0; i < test_data->D; i++) {
		double discriFunc = 0;
		for( int k = 0; k < K; k++)
			discriFunc += eta_mean[k]*Zbar_test[i][k];
		my[i] = discriFunc;
		if( discriFunc >= 0) py[i] = 1;
		else py[i] = -1;
		if( discriFunc*GEN_BIN_LABEL(test_data->doc[i].y[0], category) >= 0) acc++;
	}
	acc = (double)acc/(double)test_data->D;
	/* clean.*/
	delete sample;
	for( int i = 0; i < K; i++) {
		delete[] Ckt_test[i];
		delete[] eta_lowertriangle[i];
	}
	delete[] Ckt_test;
	delete[] Ckt_test_sum;
	delete[] eta_lowertriangle;
	return acc;
}


