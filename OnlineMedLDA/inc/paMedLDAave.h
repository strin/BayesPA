#ifndef ___paMedLDA__paMedLDAgibbs__
#define ___paMedLDA__paMedLDAgibbs__

#include <iostream>

#include "utils/utils.h"
#include "utils/InverseGaussian.h"
#include "utils/MVGaussian.h"
#include "utils/objcokus.h"
#include "Corpus.h"


namespace paMedLDA_averaging {
	template<class T>
	class Array {
	public:
		Array();
		~Array();
		
		/* create homogeneous 2d array */
		T** new_2d(int d1, int d2);
		static T** create_2d(int d1, int d2);
		/* create heterogeneous 2d array */
		T** new_2d(int d1, int* w2);
		static T** create_2d(int d1, int* w2);
		/* creae 1d array */
		T* new_1d(int d1);
		static T* create_1d(int d1);
		
		
		static void del_1d(T* array);
		static void del_2d(T** array, int d1);
		
		std::vector<T**> array2d;
		std::vector<int> array2d_dim;
		std::vector<T*>  array1d;
	};
	
	typedef struct {
		double time, ob_percent, accuracy;
		double* my;
	}Commit;

	class paMedLDAave;

	class GlobalSample {
	public:
		/* construction & destruction */
		GlobalSample(paMedLDAave* medlda);
		/* data */
		double **gamma, **prev_gamma, **stat_gamma;;
		double *gammasum; 
		double **weight_mean, **prev_weight_mean;
		double *bias, *prev_bias;	// bias term for regression.
		/* aux */
		int *stat_gamma_list_k, *stat_gamma_list_t, stat_gamma_list_end; 
		int *prev_gamma_list_k, *prev_gamma_list_t, prev_gamma_list_end;
	};

	class LocalSample {
	public:
		/* construction & destruction */
		LocalSample(paMedLDAave* medlda, CorpusData* data);
		~LocalSample();
		/* data */
		int **Z; 
		double **Zbar;
		double **Cdk, **Ckt;
		double *Ckt_sum;
		double **tau;						// solution of the dual problem.
		double **my;						// predicted label with confidence.
		double **py;						// predicted label.
		vector<int>* active_label;	        // active label queue for sparse PA.
	};

	enum PAmode {CLASSIFICATION, REGRESSION};

	class paMedLDAave {
	public:
		paMedLDAave( Corpus* corpus, int category = 1);
		~paMedLDAave();
		
		/* online BayesPA learning */
		void updateZ(vector<int>& index);
		void updateWeight(vector<int>& index, int N, bool remove);
		void inferGamma(vector<int>& index, bool reset);
		void normGamma(int N, bool remove);
		void updateZTest(int d, double& lhood, objcokus& cokus);

		/* train and inference */
		void init();
		double train(int num_iter);
		double inference( CorpusData* testData, int num_test_sample);

		/* auxiliary functions */
		void computeTheta(CorpusData* data, LocalSample* stats, int d, double* theta);
		
		/* experiment parameters */
		int m_K, m_T, m_I, m_J, m_Jburnin, m_category, m_labeln;
		double alpha, beta, train_time;
		double m_c, m_l, m_v, m_v2;
		Corpus* corpus;
		int m_epoch, m_batchsize, m_dual_steps;
		bool lets_batch, lets_multic, lets_commit, lets_bias;
		vector<int> commit_points_index;
		PAmode mode; 
		int idx;

		/* data */
		CorpusData *train_data, *test_data;

		/* stats */
		GlobalSample *global;
		LocalSample *local_train, *local_test;

		/* result */
		double test_score;
		vector<Commit> commit_points;
		double** zbar;
		
		/* source of memory */
		Array<double> mem_double;
		Array<int> mem_int;

		/* source of randomness */
		InverseGaussian invg_sampler;
		MVGaussian mvGaussian_sampler;
		objcokus cokus;

	private:
		/* auxiliary function */
		double discriminant(double* weight, double* cd, double norm);
	};

}


#endif /* defined(__OnlineTopic__GibbsSampler__) */
