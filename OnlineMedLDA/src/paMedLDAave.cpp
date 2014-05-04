#include "paMedLDAave.h"
#include "utils/utils.h"
#include "float.h"
#include <thread>

#define GEN_BIN_LABEL(y,m_category) (2*((int)(y) == (m_category))-1)
#define SIGN(val) ((double)((val) > 0)-(double)((val) < 0))

using namespace paMedLDA_averaging;

template<class T>
Array<T>::Array() {
	
}

template<class T>
Array<T>::~Array() {
	for(int i = 0; i < array1d.size(); i++) {
		del_1d(array1d[i]);
	}
	array1d.clear();
	for(int i = 0; i < array2d.size(); i++) {
		del_2d(array2d[i], array2d_dim[i]);
	}
	array2d.clear();
	array2d_dim.clear();
}

/* create homogeneous 2d array */
template<class T>
T** Array<T>::new_2d(int d1, int d2) {
	T** res = create_2d(d1,d2);
	array2d.push_back(res);
	array2d_dim.push_back(d1);
	return res;
}

template<class T>
T** Array<T>::create_2d(int d1, int d2) {
	T** res = new T*[d1];
	for(int i = 0; i < d1; i++) {
		res[i] = new T[d2];
		memset(res[i], 0, sizeof(T)*d2);
	}
	return res;
}

/* create heterogeneous 2d array */
template<class T>
T** Array<T>::new_2d(int d1, int* w2) {
	T** res = create_2d(d1, w2);
	array2d.push_back(res);
	array2d_dim.push_back(d1);
	return res;
}

template<class T>
T** Array<T>::create_2d(int d1, int* w2) {
	T** res = new T*[d1];
	for(int i = 0; i < d1; i++) {
		res[i] = new T[w2[i]];
		memset(res[i], 0, sizeof(T)*w2[i]);
	}
	return res;
}

/* creae 1d array */
template<class T>
T* Array<T>::new_1d(int d1) {
	T* res = create_1d(d1);
	array1d.push_back(res);
	return res;
}

template<class T>
T* Array<T>::create_1d(int d1) {
	T* res = new T[d1];
	memset(res, 0, sizeof(T)*d1);
	return res;
}

template<class T>
void Array<T>::del_1d(T* array) {
	delete[] array;
}


template<class T>
void Array<T>::del_2d(T** array, int d1) {
	if(array) {
		for(int i = 0; i < d1; i++) {
			if(array[i]) delete[] array[i];
		}
		delete[] array;
	}
}


/* Global Sample */
GlobalSample::GlobalSample(paMedLDAave* medlda){
	gamma = medlda->mem_double.new_2d(medlda->m_K, medlda->m_T);
	gammasum = medlda->mem_double.new_1d(medlda->m_K);
	prev_gamma = medlda->mem_double.new_2d(medlda->m_K, medlda->m_T);
	stat_gamma = medlda->mem_double.new_2d(medlda->m_K, medlda->m_T);

	weight_mean = medlda->mem_double.new_2d(medlda->m_labeln, medlda->m_K);
	prev_weight_mean = medlda->mem_double.new_2d(medlda->m_labeln, medlda->m_K);
	bias = medlda->mem_double.new_1d(medlda->m_labeln);
	prev_bias = medlda->mem_double.new_1d(medlda->m_labeln);
	
	stat_gamma_list_k = medlda->mem_int.new_1d(medlda->m_K*medlda->m_T);
	stat_gamma_list_t = medlda->mem_int.new_1d(medlda->m_K*medlda->m_T);
	prev_gamma_list_k = medlda->mem_int.new_1d(medlda->m_K*medlda->m_T);
	prev_gamma_list_t = medlda->mem_int.new_1d(medlda->m_K*medlda->m_T);
	stat_gamma_list_end = -1;
	prev_gamma_list_end = -1;
}

/* Local Sample */
LocalSample::LocalSample(paMedLDAave* medlda, CorpusData* data) {
	int W[data->D];
	for(int d = 0; d < data->D; d++) W[d] = data->doc[d].nd;
	Z = medlda->mem_int.new_2d(data->D, W);
	Zbar = medlda->mem_double.new_2d(data->D, medlda->m_K);
	Cdk = medlda->mem_double.new_2d(data->D, medlda->m_K);
	Ckt = medlda->mem_double.new_2d(medlda->m_K, medlda->m_T);
	Ckt_sum = medlda->mem_double.new_1d(medlda->m_K);
	tau = medlda->mem_double.new_2d(data->D, medlda->m_labeln);
	my = medlda->mem_double.new_2d(data->D, medlda->m_labeln);
	py = medlda->mem_double.new_2d(data->D, medlda->m_labeln);
	active_label = new vector<int>[data->D];
}

LocalSample::~LocalSample() {
	delete[] active_label;
}

/* paMedLDA-ave */
paMedLDAave::paMedLDAave(Corpus* corpus, int m_category) {
	this->corpus = corpus;
	if(corpus->multi_label == false)
		this->m_labeln = 1;
	else
		this->m_labeln = corpus->newsgroup_n;
	this->m_category = m_category;
	this->train_data = &corpus->train_data;
	this->test_data = &corpus->test_data;
	m_T									= (int)corpus->m_T;
	
	/* model parameters */
	m_K									= 5; 					// number of topics.
	m_batchsize							= 1;   					// mini-batch size.
	m_epoch								= 1;     				// number passes through the entire corpus.
	m_I									= 2;					// outer loop cycles.
	m_J									= 2;   					// inter loop cycles.
	m_Jburnin							= 0;					// inter burnin.
	m_l									= 164;					// margin control.
	m_c									= 1;					// reguarlization control.
	m_dual_steps						= 10;					// number of dual steps (for m_batchsize > 1).
	alpha								= 1/(double)m_K;  		// prior of document topic distribution.
	beta								= 0.5;			  		// prior of dictionary.
	m_v									= 1;					// prior of weight gaussian.
	lets_commit							= false;
	lets_batch							= false;
	lets_multic							= false;
	lets_bias							= false;
	mode 								= CLASSIFICATION;		// classification mode for default.

	cokus.reloadMT();
	cokus.seedMT(time(NULL)+m_category);
}

void paMedLDAave::init() {
	/* sampling setting */
	m_v2 = m_v*m_v;
	/* init stats */
	global = new GlobalSample(this);
	local_train = new LocalSample(this, train_data);
	local_test = new LocalSample(this, test_data);
	/* Initialization */
	for( int d = 0; d < train_data->D; d++) {
		memset(local_train->Cdk[d], 0, sizeof(double)*m_K);
		for( int i = 0; i < train_data->doc[d].nd; i++) {
			local_train->Z[d][i] = cokus.randomMT()%m_K;
			local_train->Cdk[d][local_train->Z[d][i]]++;
		}
	}
	memset(global->gammasum, 0, sizeof(double)*m_K);
	for( int k = 0; k < m_K; k++) {
		memset(global->stat_gamma[k], 0, sizeof(double)*m_T);
		memset(global->gamma[k], 0, sizeof(double)*m_T);
	}
	idx = 0;
	train_time = 0;
}

paMedLDAave::~paMedLDAave() {
	/* clean stats */
	delete global;
	delete local_train;
	delete local_test;
}

void paMedLDAave::updateZ(vector<int>& index) {
	CorpusData *data = train_data;
	double **Cdk = local_train->Cdk;
	int **Z = local_train->Z;
	double weights[m_K], logweights[m_K], max_logweight;
	for(auto d : index) {
		int nd = data->doc[d].nd;
		for(int i = 0; i < nd; i++) {
			int word = data->doc[d].words[i];
			Cdk[d][Z[d][i]]--;
			double cum = 0;
			if(m_labeln > 1) { // multi-task, log space.
				max_logweight = -DBL_MAX;
				for(int k = 0; k < m_K; k++) {
					logweights[k] = log(Cdk[d][k]+alpha)+log(beta+global->gamma[k][word])
												-log(beta*m_T+global->gammasum[k]);
					for(auto li : local_train->active_label[d]) {
						logweights[k] += local_train->tau[d][li]*global->weight_mean[li][k]/(double)nd;
					}
					if(logweights[k] > max_logweight)
						max_logweight = logweights[k];
				}
				for(int k = 0; k < m_K; k++) {
					weights[k] = cum+exp(logweights[k]-max_logweight);
					if(isnan(weights[k]))
						printf("error: nan weight.\n");
					cum = weights[k];
				}
			}else{             // single-task, non-log space, engenders 4X speedup.
				for(int k = 0; k < m_K; k++) {
					weights[k] = cum+(Cdk[d][k]+alpha)*(beta+global->gamma[k][word])
					            *exp(local_train->tau[d][0]*global->weight_mean[0][k]/(double)nd)
								/(beta*m_T+global->gammasum[k]);
					cum = weights[k];
				}
			}
			double sel = cum*cokus.random01();
			int seli = 0;
			for(; weights[seli] < sel; seli++);
			Z[d][i] = seli;
			Cdk[d][Z[d][i]]++; 
		}
	}
}

void paMedLDAave::updateWeight(vector<int>& index, int N, bool remove) {
	/* normalize Zbar stats */
	if(N > 1)
		for(auto d : index)
			for(int k = 0; k < m_K; k++)
				local_train->Zbar[d][k] /= N;
	/* update weight using online PA rule */
	CorpusData *data = train_data;
	double predict = 0, ell = 0;
	if(remove) {
		for(int li = 0; li < m_labeln; li++) {
			for(int k = 0; k < m_K; k++) {
				global->weight_mean[li][k] -= global->prev_weight_mean[li][k];
			}
			global->bias[li] -= global->prev_bias[li];
		}
	}
	int idx = index[0];
	if(mode == CLASSIFICATION) {
		for(int li = 0; li < m_labeln; li++) {
			if(m_batchsize == 1) {  // apply analytic PA rule.
				predict = GEN_BIN_LABEL(data->doc[idx].y[li], m_category)
				*(dotprod(local_train->Zbar[idx], global->weight_mean[li], m_K)+(lets_bias)*global->bias[li]);
				ell = max(0.0, m_l-predict);
				if(ell > 0)
					local_train->active_label[idx].push_back(li); // sparse update.
				local_train->tau[idx][li] = GEN_BIN_LABEL(data->doc[idx].y[li], m_category)
				*min(m_c, ell/((int)lets_bias+dotprod(local_train->Zbar[idx],local_train->Zbar[idx], m_K))/m_v2);
			}else{
				/* solve via gradient descent */
				vector<int> At;		// sparse gradient vector.
				double weight[m_K];
				memcpy(weight, global->weight_mean[li], sizeof(double)*m_K);
				for(int t = 1; t < m_dual_steps; t++) {
					double lrate = m_v2/t; // anneal rate ensures tau is the solution.
					At.clear();
					for(auto d : index) {
						double ell = m_l-GEN_BIN_LABEL(data->doc[d].y[li], m_category)
						*dotprod(local_train->Zbar[d], weight, m_K);
						if(ell > 0) At.push_back(d);
						local_train->tau[d][li] *= (1-lrate/m_v2);
					}
					for(int k = 0; k < m_K; k++) {
						weight[k] = weight[k]*(1-lrate/m_v2)
						+global->weight_mean[li][k]*(lrate/m_v2);
					}
					for(auto d : At) {
						for(int k = 0; k < m_K; k++) {
							weight[k] = weight[k]+m_c*lrate*GEN_BIN_LABEL(data->doc[d].y[li], m_category)*local_train->Zbar[d][k];
						}
						local_train->tau[d][li] += GEN_BIN_LABEL(data->doc[d].y[li], m_category)
						*lrate*m_c/m_v2;
					}
				}
				/* set active labels */
				for(auto d : index) {
					local_train->active_label[d].clear();
					for(int li = 0; li < m_labeln; li++)
						local_train->active_label[d].push_back(li);
				}
			}
			for(int k = 0; k < m_K; k++) {
				global->prev_weight_mean[li][k] = 0;
				for(auto d : index)
					global->prev_weight_mean[li][k] += local_train->tau[d][li]*local_train->Zbar[d][k]*m_v2;
				global->weight_mean[li][k] += global->prev_weight_mean[li][k];
			}
			if(lets_bias) {
				global->prev_bias[li] = 0;
				for(auto d : index)
					global->prev_bias[li] += local_train->tau[d][li]*m_v2;
				global->bias[li] += global->prev_bias[li];
			}
		}
	}
}

void paMedLDAave::inferGamma(vector<int>& index, bool reset) {
	CorpusData *data = train_data;
	if(reset)
		global->stat_gamma_list_end = -1;
	for(auto d : index) {
		/* reset stats for each mini-batch */
		if(reset)
			memset(local_train->Zbar[d], 0, sizeof(double)*m_K);
		for(int i = 0; i < data->doc[d].nd; i++) {
			/* cumulate Zbar stats */
			int k = local_train->Z[d][i], t = data->doc[d].words[i];
			local_train->Zbar[d][k] += 1/(double)data->doc[d].nd;
			/* update the topic dictionary, which follows a Dirichlet distribution */
			if(global->stat_gamma[k][t] == 0) {
				global->stat_gamma_list_end++;
				global->stat_gamma_list_k[global->stat_gamma_list_end] = k;
				global->stat_gamma_list_t[global->stat_gamma_list_end] = t;
			}
			global->stat_gamma[k][t]++;
		}
	}
}

void paMedLDAave::normGamma(int N, bool remove) {
	/* normalize topic dictionary */
	if(remove) {
		for(int stat_i = 0; stat_i <= global->prev_gamma_list_end; stat_i++) {
			int k = global->prev_gamma_list_k[stat_i], t = global->prev_gamma_list_t[stat_i];
			global->gamma[k][t] -= global->prev_gamma[k][t];
			global->gammasum[k] -= global->prev_gamma[k][t];
		}
	}
	global->prev_gamma_list_end = -1;
	for(int stat_i = 0; stat_i <= global->stat_gamma_list_end; stat_i++) {
		int k = global->stat_gamma_list_k[stat_i], t = global->stat_gamma_list_t[stat_i];
		global->prev_gamma[k][t] = global->stat_gamma[k][t]/(double)N;
		global->prev_gamma_list_end++;
		global->prev_gamma_list_k[global->prev_gamma_list_end] = k;
		global->prev_gamma_list_t[global->prev_gamma_list_end] = t;
		global->gamma[k][t] += global->prev_gamma[k][t];
		global->gammasum[k] += global->prev_gamma[k][t];
		global->stat_gamma[k][t] = 0;
	}
}

double paMedLDAave::discriminant(double* weight, double* cd, double norm) {
	double disc = 0;
	for( int k = 0; k < m_K; k++) {
		disc += weight[k]*cd[k];
	}
	return disc/norm;
}

void paMedLDAave::updateZTest(int d, double& lhood, objcokus& cokus) {
	double sel;
	int seli;
	double weights[m_K]; 
	Document& doc = test_data->doc[d];
	for(int i = 0; i < doc.nd; i++) {
		int t = doc.words[i];
		int zk = local_test->Z[d][i];
		local_test->Cdk[d][zk]--;
		lhood -= log((beta+global->gamma[zk][t])
								/(beta*m_T+global->gammasum[zk]));
		double cum = 0;
		for(int k = 0; k < m_K; k++) {
			weights[k] = cum+(alpha+local_test->Cdk[d][k])
								*(beta+global->gamma[k][t])
								/(beta*m_T+global->gammasum[k]);
			cum = weights[k];
		}
		sel = weights[m_K-1]*cokus.random01();
		for(seli = 0; weights[seli] < sel; seli++);
		local_test->Z[d][i] = seli;
		local_test->Cdk[d][seli]++; 
		lhood += log((beta+global->gamma[seli][t])
								/(beta*m_T+global->gammasum[seli]));
	}
}



double paMedLDAave::train(int num_iter) {
	bool lets_timing = false;
	clock_t time_start = clock();
	/* Training */
	for(int iter = 0; iter < num_iter; iter++, idx+=m_batchsize) {
		if(corpus->multi_label && iter%1000 == 0)
			printf("iter = %d K\n", iter/1000);
		/* training */
		vector<int> index;
		index.clear();
		for(int dd = idx; dd < idx+m_batchsize; dd++)
			index.push_back(dd%train_data->D);
		for(auto d : index)
			local_train->active_label[d].clear();
		for(int si = 0; si < m_I; si++) {
			for( int sj = 0; sj < m_J; sj++) {
				updateZ(index);					// Update latent assignments.
				if(sj < m_Jburnin) continue;
				inferGamma(index, sj==m_Jburnin);	// Update local stats.
			}
			normGamma(m_J-m_Jburnin, si>0);		// Normalize local stats.
			updateWeight(index, m_J-m_Jburnin, si>0);	// Update weight.
		}
	}
	/* clean */
	clock_t time_end = clock();
	train_time += (double)(time_end-time_start)/CLOCKS_PER_SEC;
	return train_time;
}

double paMedLDAave::inference(CorpusData* testData, int num_test_sample) {
	/* init */
	if(zbar) {
		Array<double>::del_2d(zbar, test_data->D);
		zbar = NULL;
	}
	zbar = Array<double>::create_2d(test_data->D, m_K);
	for(int d = 0; d < testData->D; d++) {
		memset(local_test->Z[d], 0, sizeof(int)*test_data->doc[d].nd);
		memset(local_test->Cdk[d], 0, sizeof(double)*m_K);
		for( int i = 0; i < test_data->doc[d].nd; i++) {
			local_test->Z[d][i] = cokus.randomMT()%m_K;
			local_test->Cdk[d][local_test->Z[d][i]]++;
		}
	}
	/* inference */
	int working_thread_n;
	if(m_labeln > 1)
		working_thread_n = corpus->newsgroup_n;
	else
		working_thread_n = 1;
	std::thread* threads = new std::thread[working_thread_n];
	for(int ti = 0; ti < working_thread_n; ti++) { // ti is local.
		threads[ti] = std::thread([&](int id)  {
			objcokus cokus;	// independent random space.
			cokus.seedMT(time(NULL));
			for(int d = id; d < test_data->D; d += working_thread_n) {
				/* sample */
				double lhood = 0, lhood_prev = 0;
				for(int iter = 0; iter < num_test_sample*2; iter++) {
					updateZTest(d, lhood, cokus);
					/* compute likelihood */
					double lhood_now = lhood;
					for(int k = 0; k < m_K; k++)
						lhood_now += log(sp_gamma(alpha+local_test->Cdk[d][k]));
					if(iter > 0 && fabs(lhood_now-lhood_prev) < 1e-8)
						break;
					lhood_prev = lhood_now;
				}
				for(int iter = 0; iter < num_test_sample; iter++) {
					updateZTest(d, lhood, cokus);
					for(int k = 0; k < m_K; k++) {
						zbar[d][k] += local_test->Cdk[d][k];
					}
				}
				for(int k = 0; k < m_K; k++)
					zbar[d][k] /= (double)num_test_sample;
			}		
		}, ti);
	}
	for(int ti = 0; ti < working_thread_n; ti++) threads[ti].join();
	delete[] threads;
	
	if(mode == CLASSIFICATION) {
		double acc = 0, trues = 0, pos = 0, truepos = 0;
		for(int d = 0; d < test_data->D; d++) {
			for(int li = 0; li < m_labeln; li++) {
				Document& doc = test_data->doc[d];
				double disc = discriminant(zbar[d], global->weight_mean[li], 1)+global->bias[li];
				local_test->my[d][li] = disc;
				local_test->py[d][li] = disc >= 0 ? 1 : -1;
				if(disc >= 0) trues++;
				if(doc.y[li] == 1) pos++;
				if(disc*GEN_BIN_LABEL(doc.y[li], m_category) >= 0) acc++;
				if(disc >= 0 && doc.y[li] == 1) truepos++;
			}
		}
		if(corpus->multi_label)	{		// multitask setting: F1 score.
			printf("precision = %lf, recall = %lf\n", truepos/trues, truepos/pos);
			test_score = 2/(trues/truepos+pos/truepos);
		}else							// else: accuracy.
			test_score = acc/(double)test_data->D;
	}else if(mode == REGRESSION) {
		double meany = 0;
		for(int d = 0; d < test_data->D; d++) {
			meany += test_data->doc[d].y[0];
		}
		meany /= (double)test_data->D;
		double err = 0, var = 0;
		for(int d = 0; d < test_data->D; d++) {
			for(int li = 0; li < m_labeln; li++) {
				Document& doc = test_data->doc[d];
				local_test->py[d][li] = dotprod(zbar[d], global->weight_mean[li], m_K)/doc.nd
													+(lets_bias)*global->bias[0];
				err += pow(local_test->py[d][li]-doc.y[li], 2);
				var += pow(doc.y[li]-meany, 2);
			}
		}
		test_score = 1-err/var;
	}
	return test_score;
}

