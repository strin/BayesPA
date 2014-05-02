#include <stdio.h>

#include "utils/debug.h"
#include "Corpus.h"
#include "paMedLDAave.h"

string path = "../../../data/";
string path_kaggle = "../../../../kaggle/";

using namespace paMedLDA_averaging;

void* sampler_train(void* _pamedlda) {
	paMedLDAave* pamedlda = (paMedLDAave*)_pamedlda;
	pamedlda->train();
	return NULL;
}

void* sampler_inference(void* _pamedlda) {
	paMedLDAave* pamedlda = (paMedLDAave*)_pamedlda;
	pamedlda->inference(pamedlda->test_data);
	debug( "-- pamedlda %d, acc = %lf\n", pamedlda->m_category, pamedlda->test_score);
	return NULL;
}


int main(int argc, const char * argv[])
{
	string mode = "";
	char output_file[20480] = "";
	char str_buf[20480];

	int n_runtime = 1, test_sample = 50;

	int epoch = 1, topic = 20, J = 1, I = 2, Jburnin = 0;
	int batchsize = 1;
	double m_c = 2, m_l = 16, m_beta = 0.45, m_alpha = 1, m_v = 1;
	for(int i = 1; i < argc; i++) {
		string token = string(argv[i]);
		if(token == "--binary_topic" || token == "--binary_tune"
		   || token == "--multic_topic" || token == "--multic_topic_cv"
		   || token == "--multic_commit" || token == "--multic_batchsize"
		   || token == "--regression" || token == "--multic_IJ" || token == "--multic_JBurnin"
		   || token == "--tune-regression" || token == "--multi_task"
		   || token == "--multi_task_tune" || token == "--kaggle"
		   || token == "--mtask_commit") {
			mode = string((char*)&argv[i][2]);
			strcat(output_file, mode.c_str());
		}
		if(strcmp(argv[i], "--batchsize") == 0) {
			batchsize = atoi(argv[i+1]);
			strcat(output_file, "+batchsize_");
			sprintf(str_buf, "%d", batchsize);
			strcat(output_file, str_buf);
		}
		if(strcmp(argv[i], "--Jburnin") == 0) {
			Jburnin = atoi(argv[i+1]);
			strcat(output_file, "+Jburnin_");
			sprintf(str_buf, "%d", Jburnin);
			strcat(output_file, str_buf);
		}
		if(strcmp(argv[i], "--topic") == 0) {
			topic = atoi(argv[i+1]);
			strcat(output_file, "+topic_");
			sprintf(str_buf, "%d", topic);
			strcat(output_file, str_buf);
		}
		if(strcmp(argv[i], "--c") == 0) {
			m_c = atof(argv[i+1]);
			strcat(output_file, "+c_");
			sprintf(str_buf, "%lf", m_c);
			strcat(output_file, str_buf);
		}
		if(strcmp(argv[i], "--ell") == 0) {
			m_l = atof(argv[i+1]);
			strcat(output_file, "+ell_");
			sprintf(str_buf, "%lf", m_l);
			strcat(output_file, str_buf);
		}
		if(strcmp(argv[i], "--sigma") == 0) {
			m_v = atof(argv[i+1]);
			strcat(output_file, "+sigma_");
			sprintf(str_buf, "%e", m_v);
			strcat(output_file, str_buf);
		}
		if(strcmp(argv[i], "--sigma2") == 0) {
			m_v = sqrt(atof(argv[i+1]));
			strcat(output_file, "+sigma2_");
			sprintf(str_buf, "%e", m_v*m_v);
			strcat(output_file, str_buf);
		}
		if(strcmp(argv[i], "--alpha") == 0) {
			m_alpha = atof(argv[i+1]);
			strcat(output_file, "+alpha_");
			sprintf(str_buf, "%lf", m_alpha);
			strcat(output_file, str_buf);
		}
		if(strcmp(argv[i], "--beta") == 0) {
			m_beta = atof(argv[i+1]);
			strcat(output_file, "+beta_");
			sprintf(str_buf, "%lf", m_beta);
			strcat(output_file, str_buf);
		}
		if(strcmp(argv[i], "--nruntime") == 0) {
			n_runtime = atoi(argv[i+1]);
			strcat(output_file, "+nruntime_");
			sprintf(str_buf, "%d", n_runtime);
			strcat(output_file, str_buf);
		}
		if(strcmp(argv[i], "--epoch") == 0) {
			epoch = atoi(argv[i+1]);
			strcat(output_file, "+epoch_");
			sprintf(str_buf, "%d", epoch);
			strcat(output_file, str_buf);
		}
		if(strcmp(argv[i], "--J") == 0) {
			J = atoi(argv[i+1]);
			strcat(output_file, "+J_");
			sprintf(str_buf, "%d", J);
			strcat(output_file, str_buf);
		}
		if(strcmp(argv[i], "--I") == 0) {
			I = atoi(argv[i+1]);
			strcat(output_file, "+I_");
			sprintf(str_buf, "%d", I);
			strcat(output_file, str_buf);
		}
		if(strcmp(argv[i], "--test_sample") == 0) {
			test_sample = atoi(argv[i+1]);
			strcat(output_file, "+test_sample_");
			sprintf(str_buf, "%d", test_sample);
			strcat(output_file, str_buf);
		}
		if(strcmp(argv[i], "--path") == 0) {
			path = string(argv[i+1]);
		}
	}
	strcat(output_file, ".txt");
	if(mode == "binary_topic") {
		double train_time = 0;
		FILE* fpout = fopen( output_file, "w");
		Corpus corpus;
		corpus.loadDataGML(path+"AtheismReligionMisc_Binary_train_nomalletstopwrd.gml",
							path+"AtheismReligionMisc_Binary_test_nomalletstopwrd.gml");
		double test_acc[n_runtime];
		double test_time[n_runtime];
		for( int topic = 5; topic <= 5; topic += 5) {
			for( int ni = 0; ni < n_runtime; ni++) {
				debug( "topic = %d, ni = %d\n", topic, ni);
				paMedLDAave* pamedlda = new paMedLDAave(&corpus);
				pamedlda->m_K = topic;
				pamedlda->alpha = m_alpha;
				pamedlda->m_batchsize = batchsize;
				pamedlda->m_c = m_c;
				pamedlda->m_l = m_l;
				pamedlda->m_I = I;
				pamedlda->m_J = J;
				pamedlda->m_v = m_v;
				pamedlda->m_epoch = epoch;
				pamedlda->samplen_test = test_sample;
				pamedlda->lets_batch = false;
				pamedlda->init();
				train_time = pamedlda->train();
				pamedlda->inference(pamedlda->test_data);
				printf( "[test] accuracy: %lf, train time: %lf\n", pamedlda->test_score, train_time);
				test_acc[ni] = pamedlda->test_score;
				test_time[ni] = train_time;
				delete pamedlda;
			}
			double ave_acc = vecsum(test_acc, n_runtime)/n_runtime;
			vecsubs( test_acc,  ave_acc, n_runtime);
			vecmul( test_acc, test_acc, n_runtime);
			double std_acc = sqrt(vecsum(test_acc, n_runtime)/(n_runtime-1));
			double ave_time = vecsum(test_time, n_runtime)/n_runtime;
			vecsubs( test_time,  ave_time, n_runtime);
			vecmul( test_time, test_time, n_runtime);
			double std_time = sqrt(vecsum(test_time, n_runtime)/(n_runtime-1));
			fprintf( fpout, "topic %d ave_acc %lf std_acc %lf ave_time %lf std_time %lf\n", topic, ave_acc, std_acc, ave_time, std_time);
			fprintf( stdout, "topic %d ave_acc %lf std_acc %lf ave_time %lf std_time %lf\n", topic, ave_acc, std_acc, ave_time, std_time);
			fflush(fpout);
			
		}
		fclose(fpout);
		return 0;
	}else if(mode == "binary_tune") {
		double train_time = 0;
		FILE* fpout = fopen( output_file, "w");
		Corpus corpus;
		corpus.loadDataGML(path+"AtheismReligionMisc_Binary_train_nomalletstopwrd.gml",
						   path+"AtheismReligionMisc_Binary_test_nomalletstopwrd.gml");
		double test_acc[n_runtime];
		double test_time[n_runtime];
		topic = 5;
		for(double l = 0; l <= 9; l++) {
			for(double c = -20; c <= 20; c++) {
				m_c = pow(2,c);
				m_l = pow(2,l);
				for( int ni = 0; ni < n_runtime; ni++) {
					debug( "topic = %d, ni = %d\n", topic, ni);
					paMedLDAave* pamedlda = new paMedLDAave(&corpus);
					pamedlda->m_K = topic;
					pamedlda->alpha = m_alpha;
					pamedlda->m_c = m_c;
					pamedlda->m_l = m_l;
					pamedlda->m_I = I;
					pamedlda->m_J = J;
					pamedlda->m_v = m_v;
					pamedlda->m_epoch = epoch;
					pamedlda->samplen_test = test_sample;
					pamedlda->lets_batch = false;
					pamedlda->init();
					train_time = pamedlda->train();
					pamedlda->inference(pamedlda->test_data);
					printf( "[test] accuracy: %lf, train time: %lf\n", pamedlda->test_score, train_time);
					test_acc[ni] = pamedlda->test_score;
					test_time[ni] = train_time;
					delete pamedlda;
				}
				double ave_acc = vecsum(test_acc, n_runtime)/n_runtime;
				vecsubs( test_acc,  ave_acc, n_runtime);
				vecmul( test_acc, test_acc, n_runtime);
				double std_acc = sqrt(vecsum(test_acc, n_runtime)/(n_runtime-1));
				double ave_time = vecsum(test_time, n_runtime)/n_runtime;
				vecsubs( test_time,  ave_time, n_runtime);
				vecmul( test_time, test_time, n_runtime);
				double std_time = sqrt(vecsum(test_time, n_runtime)/(n_runtime-1));
				fprintf( fpout, "m_l %lf m_c %lf ave_acc %lf std_acc %lf ave_time %lf std_time %lf\n", m_l, m_c, ave_acc, std_acc, ave_time, std_time);
				fprintf( stdout, "m_l %lf m_c %lf ave_acc %lf std_acc %lf ave_time %lf std_time %lf\n", m_l, m_c, ave_acc, std_acc, ave_time, std_time);
				fflush(fpout);
			}
		}
		fclose(fpout);
		return 0;
	}else if(mode == "multic_topic" || mode == "multic_commit" || mode == "multic_batchsize") {
		double train_time = 0, train_p_time = 0, acc;
		time_t ts_p, te_p; // parallel time.
		clock_t ts, te;    // sequential time.
		FILE* fpout = fopen( output_file, "w");
		FILE* fpres = fopen("result.txt", "a+");
		Corpus corpus;	
		corpus.loadDataGML(path+"20ng_train.gml",
							path+"20ng_test.gml");
		double test_acc[n_runtime];
		double test_time[n_runtime];
		double test_p_time[n_runtime];
		// corpus.newsgroup_n = 1;
		paMedLDAave* pamedlda[corpus.newsgroup_n];
		pthread_t threads[corpus.newsgroup_n];
		vector<int> topic_list;
		vector<int> batchsize_list;
		if(mode == "multic_topic") {
			for(int topic = 10; topic <= 110; topic += 10)
				topic_list.push_back(topic);
			batchsize_list.push_back(batchsize);
		}else if(mode == "multic_commit") {
			topic_list.push_back(40);
			batchsize_list.push_back(batchsize);
		}else if(mode == "multic_batchsize") {
			topic_list.push_back(40);
			for(int i = 0; i <= 10; i++) {
				batchsize_list.push_back(pow(2,i));
			}
		}
			
		for(auto topic : topic_list) {
			for(auto batchsize : batchsize_list) {
				for( int ni = 0; ni < n_runtime; ni++) {
					train_time = 0;
					debug("run[%d]: topic = %d, ", ni, topic);
					for( int si = 0; si < corpus.newsgroup_n; si++) {
						pamedlda[si] = new paMedLDAave(&corpus, si);
						pamedlda[si]->m_K = topic;
						pamedlda[si]->m_batchsize = batchsize;
						pamedlda[si]->lets_multic = true;
						pamedlda[si]->alpha = m_alpha;
						pamedlda[si]->beta = m_beta;
						pamedlda[si]->m_c = m_c;
						pamedlda[si]->m_l = m_l;
						pamedlda[si]->m_I = I;
						pamedlda[si]->m_J = J;
						pamedlda[si]->m_v = m_v;
						if(mode == "multic_commit" || mode == "multic_batchsize") {
							pamedlda[si]->lets_commit = true;
							int commit_point_n = 20, commit_point_spacing = 1000;
							for(int ci = 0; ci < commit_point_n; ci++) {
								pamedlda[si]->commit_points_index.push_back(ci*commit_point_spacing);
							}
						}
						pamedlda[si]->m_epoch = epoch;
						pamedlda[si]->samplen_test = test_sample;
						pamedlda[si]->lets_batch = false;
						pamedlda[si]->init();
						pthread_create(&threads[si], NULL, sampler_train, (void*)pamedlda[si]); 
					}
					time(&ts_p);
					ts = clock();
					for(int si = 0; si < corpus.newsgroup_n; si++)
						pthread_join(threads[si], NULL);
					time(&te_p);
					te = clock();
					train_time = (te-ts)/(double)CLOCKS_PER_SEC;
					train_p_time = difftime(te_p, ts_p);
					debug("training time = %lf, training time parallel = %lf", train_time, train_p_time);
					if(mode == "multic_topic") {
						for( int si = 0; si < corpus.newsgroup_n; si++) { // inference.
							pthread_create(&threads[si], NULL, sampler_inference, (void*)pamedlda[si]);
						}
						for(int si = 0; si < corpus.newsgroup_n; si++)
							pthread_join(threads[si], NULL);
						acc = 0;
						for( int d = 0; d < pamedlda[0]->test_data->D; d++) {
							int label;
							double confidence = 0-INFINITY;
							for( int si = 0; si < corpus.newsgroup_n; si++) {
								if(pamedlda[si]->local_test->my[d][0] > confidence) {
									label = si;
									confidence = pamedlda[si]->local_test->my[d][0];
								}
							}
							if(corpus.test_data.doc[d].y[0] == label) {
								acc++;
							}
						}
						test_acc[ni] = (double)acc/(double)pamedlda[0]->test_data->D;
						debug("accuracy = %lf\n", test_acc[ni]);
						fprintf( fpres, "alpha %lf beta %lf c %lf ell %lf epoch %d topic %d acc %lf", 
							m_alpha, m_beta, m_c, m_l, epoch, topic, acc);
						test_time[ni] = train_time;
						test_p_time[ni] = train_p_time;
					}else if(mode == "multic_commit" || mode == "multic_batchsize") {
						fprintf(fpout, "topic %d batchsize %d I %d J %d m_c %lf sigma %lf ", topic, batchsize, I, J, m_c, m_v);
						for(int ci = 0; ci < pamedlda[0]->commit_points_index.size(); ci++) {
							acc = 0;
							for( int d = 0; d < pamedlda[0]->test_data->D; d++) {
								int label;
								double confidence = 0-INFINITY;
								for( int si = 0; si < corpus.newsgroup_n; si++) {
									if(pamedlda[si]->commit_points[ci].my[d] > confidence) {
										label = si;
										confidence = pamedlda[si]->commit_points[ci].my[d];
									}
								}
								if(corpus.test_data.doc[d].y[0] == label) {
									acc++;
								}
							}
							fprintf(fpout, "ob %lf time %lf acc %lf ", pamedlda[0]->commit_points[ci].ob_percent, pamedlda[0]->commit_points[ci].time, (double)acc/(double)pamedlda[0]->test_data->D);
						}
						fprintf(fpout, "\n");
						fflush(fpout);
					}
					for( int si = 0; si < corpus.newsgroup_n; si++) delete pamedlda[si];
				}
				if(mode == "multic_topic") {
					double ave_acc = vecsum(test_acc, n_runtime)/n_runtime;
					vecsubs( test_acc,  ave_acc, n_runtime);
					vecmul( test_acc, test_acc, n_runtime);
					double std_acc = sqrt(vecsum(test_acc, n_runtime)/(n_runtime));
					double ave_time = vecsum(test_time, n_runtime)/n_runtime;
					vecsubs( test_time,  ave_time, n_runtime);
					vecmul( test_time, test_time, n_runtime);
					double std_time = sqrt(vecsum(test_time, n_runtime)/(n_runtime));
					double ave_p_time = vecsum(test_p_time, n_runtime)/n_runtime;
					vecsubs( test_p_time,  ave_p_time, n_runtime);
					vecmul( test_p_time, test_p_time, n_runtime);
					double std_p_time = sqrt(vecsum(test_p_time, n_runtime)/(n_runtime));
					fprintf( fpout, "topic %d ave_acc %lf std_acc %lf ave_time %lf std_time %lf ave_time_p %lf std_time_p %lf\n", topic, ave_acc, std_acc, ave_time, 
																	std_time, ave_p_time, std_p_time);
					fflush(fpout);
				}
			}
			}
		fclose(fpout);
		fclose(fpres);
	}else if(mode == "multic_IJ") {
		double train_time = 0, train_p_time = 0, acc;
		time_t ts_p, te_p; // parallel time.
		clock_t ts, te;    // sequential time.
		FILE* fpout = fopen( output_file, "w");
		FILE* fpres = fopen("result.txt", "a+");
		Corpus corpus;
		corpus.loadDataGML(path+"20ng_train.gml",
						   path+"20ng_test.gml");
		double test_acc[n_runtime];
		double test_time[n_runtime];
		double test_p_time[n_runtime];

		paMedLDAave* pamedlda[corpus.newsgroup_n];
		pthread_t threads[corpus.newsgroup_n];
		vector<int> topic_list;
		vector<int> batchsize_list;
		
		for(int I = 1; I <= 4; I++) {
			for(int J = 1; J <= 4; J++) {
				for( int ni = 0; ni < n_runtime; ni++) {
					train_time = 0;
					debug("run[%d]: topic = %d, ", ni, topic);
					for( int si = 0; si < corpus.newsgroup_n; si++) {
						pamedlda[si] = new paMedLDAave(&corpus, si);
						pamedlda[si]->m_K = topic;
						pamedlda[si]->m_batchsize = batchsize;
						pamedlda[si]->lets_multic = true;
						pamedlda[si]->alpha = m_alpha;
						pamedlda[si]->beta = m_beta;
						pamedlda[si]->m_c = m_c;
						pamedlda[si]->m_l = m_l;
						pamedlda[si]->m_I = I;
						pamedlda[si]->m_J = J;
						pamedlda[si]->m_v = m_v;
						pamedlda[si]->m_epoch = epoch;
						pamedlda[si]->samplen_test = test_sample;
						pamedlda[si]->lets_batch = false;
						pamedlda[si]->init();
						pthread_create(&threads[si], NULL, sampler_train, (void*)pamedlda[si]);
					}
					time(&ts_p);
					ts = clock();
					for(int si = 0; si < corpus.newsgroup_n; si++)
						pthread_join(threads[si], NULL);
					time(&te_p);
					te = clock();
					train_time = (te-ts)/(double)CLOCKS_PER_SEC;
					train_p_time = difftime(te_p, ts_p);
					debug("training time = %lf, training time parallel = %lf", train_time, train_p_time);

					for( int si = 0; si < corpus.newsgroup_n; si++) { // inference.
						pthread_create(&threads[si], NULL, sampler_inference, (void*)pamedlda[si]);
					}
					for(int si = 0; si < corpus.newsgroup_n; si++)
						pthread_join(threads[si], NULL);
					acc = 0;
					for( int d = 0; d < pamedlda[0]->test_data->D; d++) {
						int label;
						double confidence = 0-INFINITY;
						for( int si = 0; si < corpus.newsgroup_n; si++) {
							if(pamedlda[si]->local_test->my[d][0] > confidence) {
								label = si;
								confidence = pamedlda[si]->local_test->my[d][0];
							}
						}
						if(corpus.test_data.doc[d].y[0] == label) {
							acc++;
						}
					}
					test_acc[ni] = (double)acc/(double)pamedlda[0]->test_data->D;
					debug("accuracy = %lf\n", test_acc[ni]);
					fprintf( fpres, "alpha %lf beta %lf c %lf ell %lf epoch %d topic %d acc %lf",
							m_alpha, m_beta, m_c, m_l, epoch, topic, acc);
					test_time[ni] = train_time;
					test_p_time[ni] = train_p_time;
					for( int si = 0; si < corpus.newsgroup_n; si++) delete pamedlda[si];
				}
				double ave_acc = vecsum(test_acc, n_runtime)/n_runtime;
				vecsubs( test_acc,  ave_acc, n_runtime);
				vecmul( test_acc, test_acc, n_runtime);
				double std_acc = sqrt(vecsum(test_acc, n_runtime)/(n_runtime));
				double ave_time = vecsum(test_time, n_runtime)/n_runtime;
				vecsubs( test_time,  ave_time, n_runtime);
				vecmul( test_time, test_time, n_runtime);
				double std_time = sqrt(vecsum(test_time, n_runtime)/(n_runtime));
				double ave_p_time = vecsum(test_p_time, n_runtime)/n_runtime;
				vecsubs( test_p_time,  ave_p_time, n_runtime);
				vecmul( test_p_time, test_p_time, n_runtime);
				double std_p_time = sqrt(vecsum(test_p_time, n_runtime)/(n_runtime));
				fprintf( fpout, "topic %d batchsize %d I %d J %d ave_acc %lf std_acc %lf ave_time %lf std_time %lf ave_time_p %lf std_time_p %lf\n", topic, batchsize, I, J, ave_acc, std_acc, ave_time,
						std_time, ave_p_time, std_p_time);
				fflush(fpout);
			}
		}
		fclose(fpout);
		fclose(fpres);
	}else if(mode == "multic_JBurnin") {
		double train_time = 0, train_p_time = 0, acc;
		time_t ts_p, te_p; // parallel time.
		clock_t ts, te;    // sequential time.
		FILE* fpout = fopen( output_file, "w");
		FILE* fpres = fopen("result.txt", "a+");
		Corpus corpus;
		corpus.loadDataGML(path+"20ng_train.gml",
						   path+"20ng_test.gml");
		double test_acc[n_runtime];
		double test_time[n_runtime];
		double test_p_time[n_runtime];
		
		paMedLDAave* pamedlda[corpus.newsgroup_n];
		pthread_t threads[corpus.newsgroup_n];
		vector<int> topic_list;
		vector<int> batchsize_list;
		
		for(int J = 1; J <= 9; J+=2) {
			for(int Jburnin = 0; Jburnin < J; Jburnin+=2) {
				for( int ni = 0; ni < n_runtime; ni++) {
					train_time = 0;
					debug("run[%d]: topic = %d, ", ni, topic);
					for( int si = 0; si < corpus.newsgroup_n; si++) {
						pamedlda[si] = new paMedLDAave(&corpus, si);
						pamedlda[si]->m_K = topic;
						pamedlda[si]->m_batchsize = batchsize;
						pamedlda[si]->lets_multic = true;
						pamedlda[si]->alpha = m_alpha;
						pamedlda[si]->beta = m_beta;
						pamedlda[si]->m_c = m_c;
						pamedlda[si]->m_l = m_l;
						pamedlda[si]->m_I = I;
						pamedlda[si]->m_J = Jburnin;
						pamedlda[si]->m_J = J;
						pamedlda[si]->m_v = m_v;
						pamedlda[si]->m_epoch = epoch;
						pamedlda[si]->samplen_test = test_sample;
						pamedlda[si]->lets_batch = false;
						pamedlda[si]->init();
						pthread_create(&threads[si], NULL, sampler_train, (void*)pamedlda[si]);
					}
					time(&ts_p);
					ts = clock();
					for(int si = 0; si < corpus.newsgroup_n; si++)
						pthread_join(threads[si], NULL);
					time(&te_p);
					te = clock();
					train_time = (te-ts)/(double)CLOCKS_PER_SEC;
					train_p_time = difftime(te_p, ts_p);
					debug("training time = %lf, training time parallel = %lf", train_time, train_p_time);
					
					for( int si = 0; si < corpus.newsgroup_n; si++) { // inference.
						pthread_create(&threads[si], NULL, sampler_inference, (void*)pamedlda[si]);
					}
					for(int si = 0; si < corpus.newsgroup_n; si++)
						pthread_join(threads[si], NULL);
					acc = 0;
					for( int d = 0; d < pamedlda[0]->test_data->D; d++) {
						int label;
						double confidence = 0-INFINITY;
						for( int si = 0; si < corpus.newsgroup_n; si++) {
							if(pamedlda[si]->local_test->my[d][0] > confidence) {
								label = si;
								confidence = pamedlda[si]->local_test->my[d][0];
							}
						}
						if(corpus.test_data.doc[d].y[0] == label) {
							acc++;
						}
					}
					test_acc[ni] = (double)acc/(double)pamedlda[0]->test_data->D;
					debug("accuracy = %lf\n", test_acc[ni]);
					fprintf( fpres, "alpha %lf beta %lf c %lf ell %lf epoch %d topic %d acc %lf",
							m_alpha, m_beta, m_c, m_l, epoch, topic, acc);
					test_time[ni] = train_time;
					test_p_time[ni] = train_p_time;
					for( int si = 0; si < corpus.newsgroup_n; si++) delete pamedlda[si];
				}
				double ave_acc = vecsum(test_acc, n_runtime)/n_runtime;
				vecsubs( test_acc,  ave_acc, n_runtime);
				vecmul( test_acc, test_acc, n_runtime);
				double std_acc = sqrt(vecsum(test_acc, n_runtime)/(n_runtime));
				double ave_time = vecsum(test_time, n_runtime)/n_runtime;
				vecsubs( test_time,  ave_time, n_runtime);
				vecmul( test_time, test_time, n_runtime);
				double std_time = sqrt(vecsum(test_time, n_runtime)/(n_runtime));
				double ave_p_time = vecsum(test_p_time, n_runtime)/n_runtime;
				vecsubs( test_p_time,  ave_p_time, n_runtime);
				vecmul( test_p_time, test_p_time, n_runtime);
				double std_p_time = sqrt(vecsum(test_p_time, n_runtime)/(n_runtime));
				fprintf( fpout, "topic %d batchsize %d J %d Jburnin %d ave_acc %lf std_acc %lf ave_time %lf std_time %lf ave_time_p %lf std_time_p %lf\n", topic, batchsize, J, Jburnin, ave_acc, std_acc, ave_time,
						std_time, ave_p_time, std_p_time);
				fflush(fpout);
			}
		}
		fclose(fpout);
		fclose(fpres);
	}else if(mode == "multic_topic_cv") {
		double train_time = 0, train_p_time = 0, acc;
		time_t ts_p, te_p; // parallel time.
		clock_t ts, te;    // sequential time.
		FILE* fpout = fopen( output_file, "w");
		FILE* fpres = fopen("result.txt", "a+");
		Corpus corpus;
		corpus.loadDataGML(path+"20ng_cv_t.gml",
						   path+"20ng_cv_v.gml");
		double test_acc[n_runtime];
		double test_time[n_runtime];
		double test_p_time[n_runtime];
		paMedLDAave* pamedlda[corpus.newsgroup_n];
		pthread_t threads[corpus.newsgroup_n];
		int topic = 40;
		for(double v = 0; v <= 6; v++) {
			for(double c = 0; c <= 10; c++) {
				m_c = pow(10,c);
				m_v = sqrt(pow(10,-v));
				for( int ni = 0; ni < n_runtime; ni++) {
					train_time = 0;
					debug("multi-topic cv [%d]: topic = %d, ", ni, topic);
					for( int si = 0; si < corpus.newsgroup_n; si++) {
						pamedlda[si] = new paMedLDAave(&corpus, si);
						pamedlda[si]->m_K = topic;
						pamedlda[si]->lets_multic = true;
						pamedlda[si]->alpha = m_alpha;
						pamedlda[si]->beta = m_beta;
						pamedlda[si]->m_c = m_c;
						pamedlda[si]->m_l = m_l;
						pamedlda[si]->m_I = I;
						pamedlda[si]->m_J = J;
						pamedlda[si]->m_v = m_v;
						pamedlda[si]->m_epoch = epoch;
						pamedlda[si]->samplen_test = test_sample;
						pamedlda[si]->lets_batch = false;
						pamedlda[si]->init();
						pthread_create(&threads[si], NULL, sampler_train, (void*)pamedlda[si]);
					}
					time(&ts_p);
					ts = clock();
					for(int si = 0; si < corpus.newsgroup_n; si++)
						pthread_join(threads[si], NULL);
					time(&te_p);
					te = clock();
					train_time = (te-ts)/(double)CLOCKS_PER_SEC;
					train_p_time = difftime(te_p, ts_p);
					debug("training time = %lf, training time parallel = %lf", train_time, train_p_time);
					for( int si = 0; si < corpus.newsgroup_n; si++) { // inference.
						pthread_create(&threads[si], NULL, sampler_inference, (void*)pamedlda[si]);
					}
					for(int si = 0; si < corpus.newsgroup_n; si++)
						pthread_join(threads[si], NULL);
					acc = 0;
					
					for( int d = 0; d < pamedlda[0]->test_data->D; d++) {
						int label;
						double confidence = 0-INFINITY;
						for( int si = 0; si < corpus.newsgroup_n; si++) {
							if(pamedlda[si]->local_test->my[d][0] > confidence) {
								label = si;
								confidence = pamedlda[si]->local_test->my[d][0];
							}
						}
						if(corpus.test_data.doc[d].y[0] == label) {
							acc++;
						}
					}
					test_acc[ni] = (double)acc/(double)pamedlda[0]->test_data->D;
					debug("accuracy = %lf\n", test_acc[ni]);
					fprintf( fpres, "alpha %lf beta %lf c %lf ell %lf epoch %d topic %d acc %lf",
							m_alpha, m_beta, m_c, m_l, epoch, topic, acc);
					test_time[ni] = train_time;
					test_p_time[ni] = train_p_time;
					for( int si = 0; si < corpus.newsgroup_n; si++) delete pamedlda[si];
				}
				double ave_acc = vecsum(test_acc, n_runtime)/n_runtime;
				vecsubs( test_acc,  ave_acc, n_runtime);
				vecmul( test_acc, test_acc, n_runtime);
				double std_acc = sqrt(vecsum(test_acc, n_runtime)/(n_runtime));
				double ave_time = vecsum(test_time, n_runtime)/n_runtime;
				vecsubs( test_time,  ave_time, n_runtime);
				vecmul( test_time, test_time, n_runtime);
				double std_time = sqrt(vecsum(test_time, n_runtime)/(n_runtime));
				double ave_p_time = vecsum(test_p_time, n_runtime)/n_runtime;
				vecsubs( test_p_time,  ave_p_time, n_runtime);
				vecmul( test_p_time, test_p_time, n_runtime);
				double std_p_time = sqrt(vecsum(test_p_time, n_runtime)/(n_runtime));
				fprintf( fpout, "topic %d m_v %e m_c %e ave_acc %lf std_acc %lf ave_time %lf std_time %lf ave_time_p %lf std_time_p %lf\n", topic, m_v, m_c, ave_acc, std_acc, ave_time,
						std_time, ave_p_time, std_p_time);
				fflush(fpout);
			}
		}
		fclose(fpout);
		fclose(fpres);
	}else if(mode == "regression") {
		double train_time = 0;
		FILE* fpout = fopen( output_file, "w");
		Corpus corpus;
		corpus.loadDataGML(path+"hotelReviewTrain.gml",
						   path+"hotelReviewTest.gml");
		double test_acc[n_runtime];
		double test_time[n_runtime];
		for( int topic = 5; topic <= 30; topic += 5) {
			for( int ni = 0; ni < n_runtime; ni++) {
				debug( "topic = %d, ni = %d\n", topic, ni);
				paMedLDAave* pamedlda = new paMedLDAave(&corpus);
				pamedlda->mode = REGRESSION;
				pamedlda->m_K = topic;
				pamedlda->alpha = m_alpha;
				pamedlda->m_c = m_c;
				pamedlda->m_l = m_l;
				pamedlda->m_I = I;
				pamedlda->m_J = J;
				pamedlda->m_v = m_v;
				pamedlda->m_epoch = epoch;
				pamedlda->samplen_test = test_sample;
				pamedlda->lets_batch = false;
				pamedlda->init();
				train_time = pamedlda->train();
				pamedlda->inference(pamedlda->test_data);
				printf( "[test] accuracy: %lf, train time: %lf\n", pamedlda->test_score, train_time);
				test_acc[ni] = pamedlda->test_score;
				test_time[ni] = train_time;
				delete pamedlda;
			}
			double ave_acc = vecsum(test_acc, n_runtime)/n_runtime;
			vecsubs( test_acc,  ave_acc, n_runtime);
			vecmul( test_acc, test_acc, n_runtime);
			double std_acc = sqrt(vecsum(test_acc, n_runtime)/(n_runtime-1));
			double ave_time = vecsum(test_time, n_runtime)/n_runtime;
			vecsubs( test_time,  ave_time, n_runtime);
			vecmul( test_time, test_time, n_runtime);
			double std_time = sqrt(vecsum(test_time, n_runtime)/(n_runtime-1));
			fprintf( fpout, "topic %d ave_acc %lf std_acc %lf ave_time %lf std_time %lf\n", topic, ave_acc, std_acc, ave_time, std_time);
			fprintf( stdout, "topic %d ave_acc %lf std_acc %lf ave_time %lf std_time %lf\n", topic, ave_acc, std_acc, ave_time, std_time);
			fflush(fpout);
			
		}
		fclose(fpout);
		return 0;
	}else if(mode == "tune-regression") {
		double train_time = 0;
		FILE* fpout = fopen( output_file, "w");
		Corpus corpus;
		corpus.loadDataGML(path+"hotelReviewTrain.gml",
						   path+"hotelReviewTest.gml");
		double test_acc[n_runtime];
		double test_time[n_runtime];
		int topic = 5;
		for(double s = -10; s <= 10; s++) {
			for(double c = -10; c <= 10; c++) {
				m_c = pow(10,c);
				m_v = pow(10,s);
				for( int ni = 0; ni < n_runtime; ni++) {
					debug( "topic = %d, ni = %d\n", topic, ni);
					paMedLDAave* pamedlda = new paMedLDAave(&corpus);
					pamedlda->mode = REGRESSION;
					pamedlda->m_K = topic;
					pamedlda->alpha = m_alpha;
					pamedlda->m_c = m_c;
					pamedlda->m_l = m_l;
					pamedlda->m_I = I;
					pamedlda->m_J = J;
					pamedlda->m_v = m_v;
					pamedlda->m_epoch = epoch;
					pamedlda->samplen_test = test_sample;
					pamedlda->lets_batch = false;
					pamedlda->init();
					train_time = pamedlda->train();
					pamedlda->inference(pamedlda->test_data);
					printf( "[test] accuracy: %lf, train time: %lf\n", pamedlda->test_score, train_time);
					test_acc[ni] = pamedlda->test_score;
					test_time[ni] = train_time;
					delete pamedlda;
				}
				double ave_acc = vecsum(test_acc, n_runtime)/n_runtime;
				vecsubs( test_acc,  ave_acc, n_runtime);
				vecmul( test_acc, test_acc, n_runtime);
				double std_acc = sqrt(vecsum(test_acc, n_runtime)/(n_runtime-1));
				double ave_time = vecsum(test_time, n_runtime)/n_runtime;
				vecsubs( test_time,  ave_time, n_runtime);
				vecmul( test_time, test_time, n_runtime);
				double std_time = sqrt(vecsum(test_time, n_runtime)/(n_runtime-1));
				fprintf( fpout, "m_c %lf m_v %lf m_l %lf topic %d ave_acc %lf std_acc %lf ave_time %lf std_time %lf\n", m_c, m_v, m_l, topic, ave_acc, std_acc, ave_time, std_time);
				fprintf( stdout, "m_c %lf m_v %lf m_l %lf topic %d ave_acc %lf std_acc %lf ave_time %lf std_time %lf\n", m_c, m_v, m_l, topic, ave_acc, std_acc, ave_time, std_time);
				fflush(fpout);
			}
		}
		fclose(fpout);
		return 0;
	}
	else if(mode == "multi_task" || mode == "mtask_commit") {
		double train_time = 0;
		FILE* fpout = fopen( output_file, "w");
		Corpus corpus;
		corpus.loadDataGML(path+"wiki_train.gml",
						   path+"wiki_test.gml", true);
		double test_acc[n_runtime];
		double test_time[n_runtime];
		for( int topic = 40; topic <= 40; topic += 5) {
			for( int ni = 0; ni < n_runtime; ni++) {
				debug( "topic = %d, ni = %d\n", topic, ni);
				paMedLDAave* pamedlda = new paMedLDAave(&corpus);
				pamedlda->m_K = topic;
				pamedlda->alpha = m_alpha;
				pamedlda->beta = m_beta;
				pamedlda->m_batchsize = batchsize;
				pamedlda->m_c = m_c;
				pamedlda->m_l = m_l;
				pamedlda->m_I = I;
				pamedlda->m_J = J;
				pamedlda->m_v = m_v;
				pamedlda->m_epoch = epoch;
				pamedlda->samplen_test = test_sample;
				if(mode == "mtask_commit") {
					pamedlda->lets_commit = true;
					int commit_point_n = ceil(log(1100000)/log(2));
					for(int ci = ceil(log(batchsize)/log(2)); ci <= commit_point_n; ci++) {
						pamedlda->commit_points_index.push_back(pow(2,ci));
					}
				}
				pamedlda->lets_batch = false;
				pamedlda->init();
				train_time = pamedlda->train();
//				pamedlda->inference(pamedlda->test_data);
				printf( "[test] accuracy: %lf, train time: %lf\n", pamedlda->test_score, train_time);
				test_acc[ni] = pamedlda->test_score;
				test_time[ni] = train_time;
				if(pamedlda->lets_commit) {
					fprintf(fpout, "topic %d batchsize 1 ", topic);
					for(int ci = 0; ci < pamedlda->commit_points.size(); ci++) {
						Commit& commit = pamedlda->commit_points[ci];
						fprintf(fpout, "ob %lf time %lf f1 %lf ", commit.ob_percent, commit.time, commit.accuracy);
					}
					fflush(fpout);
				}
				delete pamedlda;
			}
			if(mode == "multi_task") {
				double ave_acc = vecsum(test_acc, n_runtime)/n_runtime;
				vecsubs( test_acc,  ave_acc, n_runtime);
				vecmul( test_acc, test_acc, n_runtime);
				double std_acc = sqrt(vecsum(test_acc, n_runtime)/(n_runtime-1));
				double ave_time = vecsum(test_time, n_runtime)/n_runtime;
				vecsubs( test_time,  ave_time, n_runtime);
				vecmul( test_time, test_time, n_runtime);
				double std_time = sqrt(vecsum(test_time, n_runtime)/(n_runtime-1));
				fprintf( fpout, "topic %d ave_acc %lf std_acc %lf ave_time %lf std_time %lf\n", topic, ave_acc, std_acc, ave_time, std_time);
				fprintf( stdout, "topic %d ave_acc %lf std_acc %lf ave_time %lf std_time %lf\n", topic, ave_acc, std_acc, ave_time, std_time);
				fflush(fpout);
			}
		}
		fclose(fpout);
		return 0;
	}else if(mode == "multi_task_tune") {
		double train_time = 0;
		FILE* fpout = fopen( output_file, "w");
		Corpus corpus;
		corpus.loadDataGML(path+"wiki_train.gml",
						   path+"wiki_test.gml", true);
		double test_acc[n_runtime];
		double test_time[n_runtime];
		topic = 40;
		for(double v = 0; v <= 5; v++) {
			for(double c = -10; c <= 1; c++) {
				m_c = pow(3,c);
				m_v = pow(10, v);
				for( int ni = 0; ni < n_runtime; ni++) {
					debug( "topic = %d, ni = %d\n", topic, ni);
					paMedLDAave* pamedlda = new paMedLDAave(&corpus);
					pamedlda->m_K = topic;
					pamedlda->alpha = m_alpha;
					pamedlda->beta = m_beta;
					pamedlda->m_c = m_c;
					pamedlda->m_l = m_l;
					pamedlda->m_I = I;
					pamedlda->m_J = J;
					pamedlda->m_v = m_v;
					pamedlda->m_epoch = epoch;
					pamedlda->samplen_test = test_sample;
					pamedlda->lets_batch = false;
					pamedlda->init();
					train_time = pamedlda->train();
					pamedlda->inference(pamedlda->test_data);
					printf( "[test] accuracy: %lf, train time: %lf\n", pamedlda->test_score, train_time);
					test_acc[ni] = pamedlda->test_score;
					test_time[ni] = train_time;
					delete pamedlda;
				}
				double ave_acc = vecsum(test_acc, n_runtime)/n_runtime;
				vecsubs( test_acc,  ave_acc, n_runtime);
				vecmul( test_acc, test_acc, n_runtime);
				double std_acc = sqrt(vecsum(test_acc, n_runtime)/(n_runtime-1));
				double ave_time = vecsum(test_time, n_runtime)/n_runtime;
				vecsubs( test_time,  ave_time, n_runtime);
				vecmul( test_time, test_time, n_runtime);
				double std_time = sqrt(vecsum(test_time, n_runtime)/(n_runtime-1));
				fprintf( fpout, "m_c %lf m_l %lf ave_acc %lf std_acc %lf ave_time %lf std_time %lf\n", m_c, m_l, ave_acc, std_acc, ave_time, std_time);
				fprintf( stdout, "m_c %lf m_l %lf ave_acc %lf std_acc %lf ave_time %lf std_time %lf\n", m_c, m_l, ave_acc, std_acc, ave_time, std_time);
				fflush(fpout);
				
			}
		}
		fclose(fpout);
		return 0;
	}
    return 0;
}

