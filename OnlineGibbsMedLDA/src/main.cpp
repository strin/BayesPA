//
//  main.c
//  OnlineTopic
//
//  Created by Tianlin Shi on 4/29/13.
//  Copyright (c) 2013 Tianlin Shi. All rights reserved.
//

#include <stdio.h>

#include "debug.h"
#include "Setting.h"
#include "Corpus.h"
#include "Mapper.h"

#include "MVGaussian.h"
#include "cholesky.h"
#include "ap.h"
#include "apaux.h"
#include "paMedLDA-gibbs.h"
// #include "paMedLDAgibbsmt.h"

string path = "../../../../data/";

void* sampler_train(void* _sampler) {
	paMedLDAgibbs* sampler = (paMedLDAgibbs*)_sampler;
	sampler->train();
}

void* sampler_inference(void* _sampler) {
	paMedLDAgibbs* sampler = (paMedLDAgibbs*)_sampler;
	double acc = sampler->inference(sampler->test_data);
	debug( "-- sampler %d, acc = %lf\n", sampler->category, acc);
}


int main(int argc, const char * argv[])
{
	string mode = "";
	int n_runtime = 10, test_sample = 200;
	double m_c = 5, m_l = 1, m_alpha = 0.8, m_beta = 0.5;
	int epoch = 10, topic = 20, J = 3, I = 1;
	char output_file[20480] = "";
	char str_buf[20480];
	int batchsize = 32;
	for(int i = 1; i < argc; i++) {
		string token = string(argv[i]);
		if(token == "--binary_topic" || token == "--multic_topic"
		|| token == "--multic_commit" || token == "--mtask_commit"
		|| token == "--multic_ij" || token == "--multic_bj") {
			mode = string((char*)&argv[i][2]);
			strcat(output_file, mode.c_str());
		}
		if(strcmp(argv[i], "--batchsize") == 0) {
			batchsize = atoi(argv[i+1]);
			strcat(output_file, "+batchsize_");
			sprintf(str_buf, "%d", batchsize);
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
	Corpus corpus;
	if(mode == "binary_topic") {
		double train_time = 0;
		FILE* fpout = fopen( output_file, "w");
		corpus.loadDataGML(path+"AtheismReligionMisc_Binary_train_nomalletstopwrd.gml",
							path+"AtheismReligionMisc_Binary_test_nomalletstopwrd.gml");
		//	corpus.loadDataGML("data/20ng_train.gml",
		//						"data/20ng_test.gml");
		//	for( int i =0; i < corpus.trainDataSize; i++)
		//		corpus.trainData[i]->label = 2*(corpus.trainData[i]->label==0)-1;
		//	for( int i =0; i < corpus.testDataSize; i++)
		//		corpus.test_data[i]->label = 2*(corpus.test_data[i]->label==0)-1;
		double* test_acc = new double[n_runtime];
		double* test_time = new double[n_runtime];
		for( int k_topic = 5; k_topic <= 30; k_topic += 5) {
			for( int ni = 0; ni < n_runtime; ni++) {
				debug( "k_topic = %d, ni = %d\n", k_topic, ni);
				paMedLDAgibbs* sampler = new paMedLDAgibbs(&corpus);
				sampler->K = k_topic;
				sampler->I = I;
				sampler->J = J;
				sampler->batchSize = 32;
				sampler->epoch = epoch;
				sampler->max_gibbs_iter = test_sample;
				sampler->lets_batch = false;
				sampler->init();
				train_time = sampler->train();
				sampler->test_acc = sampler->inference(sampler->test_data);
				printf( "[test] accuracy: %lf, train time: %lf\n", sampler->test_acc, train_time);
				test_acc[ni] = sampler->test_acc;
				test_time[ni] = train_time;
				delete sampler;
			}
			double ave_acc = vecsum(test_acc, n_runtime)/n_runtime;
			vecsubs( test_acc,  ave_acc, n_runtime);
			vecmul( test_acc, test_acc, n_runtime);
			double std_acc = sqrt(vecsum(test_acc, n_runtime)/(n_runtime-1));
			double ave_time = vecsum(test_time, n_runtime)/n_runtime;
			vecsubs( test_time,  ave_time, n_runtime);
			vecmul( test_time, test_time, n_runtime);
			double std_time = sqrt(vecsum(test_time, n_runtime)/(n_runtime-1));
			fprintf( fpout, "topic %d ave_acc %lf std_acc %lf ave_time %lf std_time %lf\n", k_topic, ave_acc, std_acc, ave_time, std_time);
			fprintf( stdout, "topic %d ave_acc %lf std_acc %lf ave_time %lf std_time %lf\n", k_topic, ave_acc, std_acc, ave_time, std_time);
			fflush(fpout);
		}
		fclose(fpout);
		delete[] test_acc;
		delete[] test_time;
		return 0;
	}else if(mode == "multic_topic") {
		int n_runtime = 1;
		double train_time = 0, train_p_time = 0, acc;
		time_t ts_p, te_p; // parallel time.
		clock_t ts, te;    // sequential time.
		FILE* fpout = fopen( output_file, "w");
		FILE* fpres = fopen("result.txt", "a+");
		corpus.loadDataGML(path+"20ng_train.gml",
							path+"20ng_test.gml");
		double* test_acc = new double[n_runtime];
		double* test_time = new double[n_runtime];
		double* test_p_time = new double[n_runtime];
		paMedLDAgibbs* sampler[corpus.newsgroup_n];
		pthread_t threads[corpus.newsgroup_n];
		for( int k_topic = 20; k_topic <= 20; k_topic += 5) {
			for( int ni = 0; ni < n_runtime; ni++) {
				train_time = 0;
				debug("run[%d]: k_topic = %d, ", ni, k_topic);
				for( int si = 0; si < corpus.newsgroup_n; si++) {
					sampler[si] = new paMedLDAgibbs(&corpus, si);
					sampler[si]->batchSize = batchsize;
					sampler[si]->K = k_topic;
					sampler[si]->lets_multic = true;
					sampler[si]->init();
					pthread_create(&threads[si], NULL, sampler_train, (void*)sampler[si]); // train.
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
					pthread_create(&threads[si], NULL, sampler_inference, (void*)sampler[si]);
				}
				for(int si = 0; si < corpus.newsgroup_n; si++)
					pthread_join(threads[si], NULL);
				acc = 0;
				
				for( int i = 0; i < sampler[0]->test_data->D; i++) {
					int label;
					double confidence = 0-INFINITY;
					for( int si = 0; si < corpus.newsgroup_n; si++) {
						if( sampler[si]->my[i] > confidence) {
							label = si;
							confidence = sampler[si]->my[i];
						}
					}
					if(label == corpus.test_data.doc[i].y[0]) {
						acc++;
					}
				}
				test_acc[ni] = (double)acc/(double)sampler[0]->test_data->D;
				debug("accuracy = %lf\n", test_acc[ni]);
				fprintf( fpres, "alpha %lf beta %lf c %lf ell %lf epoch %d topic %d acc %lf", m_alpha, m_beta, m_c, m_l, epoch, topic, acc);
				test_time[ni] = train_time;
				test_p_time[ni] = train_p_time;
				for( int si = 0; si < corpus.newsgroup_n; si++) delete sampler[si];
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
			fprintf( fpout, "topic %d ave_acc %lf std_acc %lf ave_time %lf std_time %lf ave_time_p %lf std_time_p %lf\n", k_topic, ave_acc, std_acc, ave_time, std_time, ave_p_time, std_p_time);
			fflush(fpout);
		}
		fclose(fpout);
		fclose(fpres);
		delete[] test_acc;
		delete[] test_time;
	}else if(mode == "multic_epoch") {
		double train_time = 0, train_p_time = 0, acc;
		time_t ts_p, te_p; // parallel time.
		clock_t ts, te;    // sequential time.
		FILE* fpout = fopen( output_file, "w");
		corpus.loadDataGML(path+"20ng_train.gml",
							path+"20ng_test.gml");
		double* test_acc = new double[n_runtime];
		double* test_time = new double[n_runtime];
		double* test_p_time = new double[n_runtime];
		paMedLDAgibbs* sampler[corpus.newsgroup_n];
		pthread_t threads[corpus.newsgroup_n];
		for( int m_epoch = 0; m_epoch <= 20; m_epoch++) {
			for( int ni = 0; ni < n_runtime; ni++) {
				train_time = 0;
				for( int si = 0; si < corpus.newsgroup_n; si++) {
					sampler[si] = new paMedLDAgibbs(&corpus, si);
					sampler[si]->batchSize = corpus.train_data.D;
					sampler[si]->epoch = m_epoch;
					sampler[si]->K = topic;
					sampler[si]->lets_multic = true;
					sampler[si]->init();
					pthread_create(&threads[si], NULL, sampler_train, (void*)sampler[si]); // train.
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
					pthread_create(&threads[si], NULL, sampler_inference, (void*)sampler[si]);
				}
				for(int si = 0; si < corpus.newsgroup_n; si++)
					pthread_join(threads[si], NULL);
				acc = 0;
				
				for( int i = 0; i < sampler[0]->test_data->D; i++) {
					int label;
					double confidence = 0-INFINITY;
					for( int si = 0; si < corpus.newsgroup_n; si++) {
						if( sampler[si]->my[i] > confidence) {
							label = si;
							confidence = sampler[si]->my[i];
						}
					}
					if(label == corpus.test_data.doc[i].y[0]) {
						acc++;
					}
				}
				test_acc[ni] = (double)acc/(double)sampler[0]->test_data->D;
				debug("accuracy = %lf\n", test_acc[ni]);
				test_time[ni] = train_time;
				test_p_time[ni] = train_p_time;
				for( int si = 0; si < corpus.newsgroup_n; si++) delete sampler[si];
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
			fprintf( fpout, "epoch %d ave_acc %lf std_acc %lf ave_time %lf std_time %lf ave_time_p %lf std_time_p %lf\n", m_epoch, ave_acc, std_acc, ave_time, std_time, ave_p_time, std_p_time);
			fflush(fpout);
		}
		fclose(fpout);
		delete[] test_acc;
		delete[] test_time;
	}
	/*else if(mode == "mtask_commit") {
		int n_runtime = 1;
		double train_p_time = 0, acc;
		time_t ts_p, te_p; // parallel time.
		FILE* fpout = fopen( output_file, "w");
		corpus.loadDataGML(path+"wiki_train_subset.gml",
							   path+"wiki_test.gml", true);
		double* test_acc = new double[n_runtime];
		double* test_p_time = new double[n_runtime];
		paMedLDAgibbsmt* sampler;
		pthread_t threads[corpus.newsgroup_n];
		for( int k_topic = 20; k_topic <= 20; k_topic += 5) {
			for( int ni = 0; ni < n_runtime; ni++) {
				debug("run[%d]: k_topic = %d, ", ni, k_topic);
				sampler = new paMedLDAgibbsmt(corpus, corpus.newsgroup_n);
				sampler->batchSize = batchsize;
				sampler->K = k_topic;
				sampler->m_c = m_c;
				sampler->m_l = m_l;
				sampler->J = J;
				sampler->alpha0 = m_alpha;
				sampler->beta0 = m_beta;
				sampler->lets_commit = true;
				sampler->commit_point_n = 10;
				sampler->commit_point_spacing = (1+sampler->data->D/sampler->batchSize)/sampler->commit_point_n+1;
				printf("# commit points %d, spacing %d\n", sampler->commit_point_n, sampler->commit_point_spacing);
				sampler->init();
				time(&ts_p);
				sampler->train();
				time(&te_p);
				train_p_time = difftime(te_p, ts_p);
				debug("training time = %lf", train_p_time);
				double f1 = sampler->inference(sampler->test_data);
				test_acc[ni] = f1;
				debug("f1 score = %lf\n", test_acc[ni]);
				test_p_time[ni] = train_p_time;
				fprintf( fpout, "topic %d ", k_topic);
				for(int ci = 0; ci < sampler->commit_points.size(); ci++) {
					fprintf(fpout, "ob %lf time %lf acc %lf ", sampler->commit_points[ci].ob_percent, sampler->commit_points[ci].time, sampler->commit_points[ci].accuracy);
				}
				fprintf(fpout, "\n");
				delete sampler;
			}
			fflush(fpout);
		}
		fclose(fpout);
		delete[] test_acc;
		delete[] test_p_time;
	}*/else if(mode == "multic_commit") {
		FILE* fpout = fopen( output_file, "w");
		corpus.loadDataGML(path+"20ng_train.gml",
							path+"20ng_test.gml");
		paMedLDAgibbs* sampler[corpus.newsgroup_n];
		pthread_t threads[corpus.newsgroup_n];
		int cps[5] = {500, 125, 32, 8, 2};
		int cpn[5] = {10, 10, 10, 10, 10};
		for(int i = 0; i <= 4; i++ )  {
			for( int ni = 0; ni < n_runtime; ni++) {
				for( int si = 0; si < corpus.newsgroup_n; si++) {
					sampler[si] = new paMedLDAgibbs(&corpus, si);
					if(topic == -1) {
						sampler[si]->K = i*10+10;
					}else{
						sampler[si]->K = topic;
					}
					if(batchsize == -1) {
						sampler[si]->batchSize = pow(2,(i+1)*2);
					}else{
						sampler[si]->batchSize = batchsize;
					}
					sampler[si]->lets_multic = true;
					sampler[si]->lets_commit = true;
					sampler[si]->commit_point_spacing = cps[i];
					sampler[si]->commit_point_n = cpn[i];
					sampler[si]->init();
					pthread_create(&threads[si], NULL, sampler_train, (void*)sampler[si]); // train.
				}
				for(int si = 0; si < corpus.newsgroup_n; si++)
					pthread_join(threads[si], NULL);
				for( int si = 0; si < corpus.newsgroup_n; si++) { // inference.
					pthread_create(&threads[si], NULL, sampler_inference, (void*)sampler[si]);
				}
				for(int si = 0; si < corpus.newsgroup_n; si++)
					pthread_join(threads[si], NULL);
				fprintf( fpout, "topic %d batchsize %d J %d ", sampler[0]->K, sampler[0]->batchSize, sampler[0]->J);
				for( int ci = 0; ci < sampler[0]->commit_points.size(); ci++) {
					int acc = 0, label = 0;
					for( int i = 0; i < sampler[0]->test_data->D; i++) {
						int label;
						double confidence = 0-INFINITY;
						for( int si = 0; si < corpus.newsgroup_n; si++) {
							if( sampler[si]->commit_points.at(ci).my[i] > confidence) {
								label = si;
								confidence = sampler[si]->commit_points.at(ci).my[i];
							}
						}
						if(label == corpus.test_data.doc[i].y[0]) {
							acc++;
						}
					}
					fprintf( fpout, "ob %lf time %lf acc %lf ", sampler[0]->commit_points.at(ci).ob_percent,
							sampler[0]->commit_points.at(ci).time,
							(double)acc/sampler[0]->test_data->D);
					for(int si = 0; si < corpus.newsgroup_n; si++) {
						delete[] sampler[si]->commit_points.at(ci).my;
					}
				}
				fprintf( fpout, "\n");
				for( int si = 0; si < corpus.newsgroup_n; si++) delete sampler[si];
			}
		}
		fclose(fpout);
	}
	if(mode == "multic_ij") {
		double train_time = 0, train_p_time = 0, acc;
		time_t ts_p, te_p; // parallel time.
		clock_t ts, te;    // sequential time.
		FILE* fpout = fopen( output_file, "w");
		FILE* fpres = fopen("result.txt", "a+");
		corpus.loadDataGML(path+"20ng_train.gml",
							path+"20ng_test.gml");
		double* test_acc = new double[n_runtime];
		double* test_time = new double[n_runtime];
		double* test_p_time = new double[n_runtime];
		paMedLDAgibbs* sampler[corpus.newsgroup_n];
		pthread_t threads[corpus.newsgroup_n];
		for(int I = 1; I <= 4; I++) {
			for(int J = 1; J <= 5; J++) {
				for( int ni = 0; ni < n_runtime; ni++) {
					train_time = 0;
					debug("run[%d]: k_topic = %d, ", ni, topic);
					for( int si = 0; si < corpus.newsgroup_n; si++) {
						sampler[si] = new paMedLDAgibbs(&corpus, si);
						sampler[si]->batchSize = batchsize;
						sampler[si]->K = topic;
						sampler[si]->I = I;
						sampler[si]->J = J;
						sampler[si]->lets_multic = true;
						sampler[si]->init();
						pthread_create(&threads[si], NULL, sampler_train, (void*)sampler[si]); // train.
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
						pthread_create(&threads[si], NULL, sampler_inference, (void*)sampler[si]);
					}
					for(int si = 0; si < corpus.newsgroup_n; si++)
						pthread_join(threads[si], NULL);
					acc = 0;
					
					for( int i = 0; i < sampler[0]->test_data->D; i++) {
						int label;
						double confidence = 0-INFINITY;
						for( int si = 0; si < corpus.newsgroup_n; si++) {
							if( sampler[si]->my[i] > confidence) {
								label = si;
								confidence = sampler[si]->my[i];
							}
						}
						if(label == corpus.test_data.doc[i].y[0]) {
							acc++;
						}
					}
					test_acc[ni] = (double)acc/(double)sampler[0]->test_data->D;
					debug("accuracy = %lf\n", test_acc[ni]);
					fprintf( fpres, "alpha %lf beta %lf c %lf ell %lf epoch %d topic %d acc %lf", m_alpha, m_beta, m_c, m_l, epoch, topic, acc);
					test_time[ni] = train_time;
					test_p_time[ni] = train_p_time;
					for( int si = 0; si < corpus.newsgroup_n; si++) delete sampler[si];
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
				fprintf( fpout, "topic %d I %d J %d ave_acc %lf std_acc %lf ave_time %lf std_time %lf ave_time_p %lf std_time_p %lf\n", topic, I, J, ave_acc, std_acc, ave_time, std_time, ave_p_time, std_p_time);
				fflush(fpout);
			}
		}
		fclose(fpout);
		fclose(fpres);
		delete[] test_acc;
		delete[] test_time;
	}
    return 0;
}

