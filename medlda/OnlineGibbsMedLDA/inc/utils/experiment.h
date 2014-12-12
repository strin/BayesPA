//
//  experiment.h
//  OnlineTopic
//
//  Created by Tianlin Shi on 7/25/13.
//  Copyright (c) 2013 Tianlin Shi. All rights reserved.
//

#ifndef __OnlineTopic__experiment__
#define __OnlineTopic__experiment__

#include "debug.h"
#include "Setting.h"
#include "Corpus.h"
#include "Mapper.h"

#include "MVGaussian.h"
#include "cholesky.h"
#include "ap.h"
#include "apaux.h"

Corpus* corpus;
Setting* setting;

string path = "../../../data/";

void binary_batch_topic() {
	int n_runtime = 10;
	double train_time = 0;
	FILE* fpout = fopen( "binary_batch_topic.txt", "w");
	corpus = new Corpus();
	corpus->dic = corpus->loadDictionary(path+"dic.txt");
	corpus->loadDataGML(path+"AtheismReligionMisc_Binary_train_nomalletstopwrd.gml",
						path+"AtheismReligionMisc_Binary_test_nomalletstopwrd.gml");
	double* test_acc = new double[n_runtime];
	double* test_time = new double[n_runtime];
	for( int k_topic = 5; k_topic <= 30; k_topic += 5) {
		
		for( int ni = 0; ni < n_runtime; ni++) {
			debug( "k_topic = %d, ni = %d\n", k_topic, ni);
			HybridMedLDA* sampler = new HybridMedLDA(corpus);
			sampler->K = k_topic;
			sampler->init();
			train_time = sampler->train();
			sampler->test_acc = sampler->inference(sampler->testData);
			debug( "[test] accuracy: %lf, train time: %lf\n", sampler->test_acc, train_time);
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
		fflush(fpout);
	}
	fclose(fpout);
	delete[] test_acc;
	delete[] test_time;
}

void binary_online_topic() {
	int n_runtime = 10;
	double train_time = 0;
	FILE* fpout = fopen( "hybrid_binary_online_topic.txt", "w");
	corpus = new Corpus();
	corpus->dic = corpus->loadDictionary(path+"dic.txt");
	corpus->loadDataGML(path+"AtheismReligionMisc_Binary_train_nomalletstopwrd.gml",
						path+"AtheismReligionMisc_Binary_test_nomalletstopwrd.gml");
	double* test_acc = new double[n_runtime];
	double* test_time = new double[n_runtime];
	for( int k_topic = 5; k_topic <= 25; k_topic += 5) {
		for( int ni = 0; ni < n_runtime; ni++) {
			debug( "k_topic = %d, ni = %d\n", k_topic, ni);
			HybridMedLDA* sampler = new HybridMedLDA(corpus);
			sampler->K = k_topic;
			sampler->batchSize = 64;
			sampler->init();
			train_time = sampler->train();
			sampler->test_acc = sampler->inference(sampler->testData);
			debug( "[test] accuracy: %lf, train time: %lf\n", sampler->test_acc, train_time);
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
		debug("topic %d ave_acc %lf std_acc %lf ave_time %lf std_time %lf\n", k_topic, ave_acc, std_acc, ave_time, std_time);
		fprintf( fpout, "topic %d ave_acc %lf std_acc %lf ave_time %lf std_time %lf\n", k_topic, ave_acc, std_acc, ave_time, std_time);
		fflush(fpout);
	}
	fclose(fpout);
	delete[] test_acc;
	delete[] test_time;
}

void binary_online_commit() {
	int n_runtime = 10;
	double train_time = 0;
	FILE* fpout = fopen( "hybrid_binary_online_commit.txt", "w");
	corpus = new Corpus();
	corpus->dic = corpus->loadDictionary(path+"dic.txt");
	corpus->loadDataGML(path+"AtheismReligionMisc_Binary_train_nomalletstopwrd.gml",
						path+"AtheismReligionMisc_Binary_test_nomalletstopwrd.gml");
	double* test_acc = new double[n_runtime];
	double* test_time = new double[n_runtime];
	for(int bsize = 0; bsize <= 7; bsize++) {
		for( int k_topic = 5; k_topic <= 30; k_topic += 5) {
			for( int ni = 0; ni < n_runtime; ni++) {
				HybridMedLDA* sampler = new HybridMedLDA(corpus);
				sampler->K = k_topic;
				sampler->batchSize = pow(2,bsize);
				sampler->epoch = 3;
				sampler->commit_point_spacing = corpus->trainDataSize/(5*sampler->batchSize);
				sampler->lets_commit = true;
				sampler->init();
				debug( "k_topic = %d, ni = %d, batchsize = %d\n", k_topic, ni, sampler->batchSize);
				train_time = sampler->train();
				sampler->test_acc = sampler->inference(sampler->testData);
				debug( "[test] accuracy: %lf, train time: %lf\n", sampler->test_acc, train_time);
				test_acc[ni] = sampler->test_acc;
				test_time[ni] = train_time;
				fprintf( fpout, "topic %d batchsize %d spacing %d ", k_topic, sampler->batchSize, sampler->commit_point_spacing);
				for(int ci = 0; ci < sampler->commit_points.size(); ci++) {
					fprintf( fpout, "ob %lf time %lf acc %lf ", sampler->commit_points[ci].ob_percent,
							sampler->commit_points[ci].time, sampler->commit_points[ci].accuracy);
				}
				fprintf( fpout, "\n");
				fflush(fpout);
				delete sampler;
			}
		}
	}
	fclose(fpout);
	delete[] test_acc;
	delete[] test_time;
}


void binary_online_tune_prior() {
	int n_runtime = 10;
	double train_time = 0;
	FILE* fpout = fopen( "binary_batch_topic.txt", "w");
	corpus = new Corpus();
	corpus->dic = corpus->loadDictionary(path+"dic.txt");
	corpus->loadDataGML(path+"AtheismReligionMisc_Binary_train_nomalletstopwrd.gml",
						path+"AtheismReligionMisc_Binary_test_nomalletstopwrd.gml");
	double* test_acc = new double[n_runtime];
	double* test_time = new double[n_runtime];
	for(double beta = 0.1; beta <= 0.5; beta += 0.05) {
		for( int k_topic = 5; k_topic <= 25; k_topic += 20) {
			for( int ni = 0; ni < n_runtime; ni++) {
				debug( "beta = %lf k_topic = %d, ni = %d\n", beta, k_topic, ni);
				HybridMedLDA* sampler = new HybridMedLDA(corpus);
				sampler->K = k_topic;
				sampler->batchSize = 64;
				sampler->beta0 = beta;
				sampler->init();
				train_time = sampler->train();
				sampler->test_acc = sampler->inference(sampler->testData);
				debug( "[test] accuracy: %lf, train time: %lf\n", sampler->test_acc, train_time);
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
			debug("beta %lf topic %d ave_acc %lf std_acc %lf ave_time %lf std_time %lf\n", beta, k_topic, ave_acc, std_acc, ave_time, std_time);
			fprintf( fpout, "beta %lf topic %d ave_acc %lf std_acc %lf ave_time %lf std_time %lf\n", beta, k_topic, ave_acc, std_acc, ave_time, std_time);
			fflush(fpout);
		}
	}
	fclose(fpout);
	delete[] test_acc;
	delete[] test_time;
}

void* sampler_train(void* _sampler) {
	HybridMedLDA* sampler = (HybridMedLDA*)_sampler;
	sampler->train();
}

void* sampler_inference(void* _sampler) {
	HybridMedLDA* sampler = (HybridMedLDA*)_sampler;
	double acc = sampler->inference(sampler->testData);
	debug( "-- sampler %d, acc = %lf\n", sampler->category, acc);
}

void multic_batch_topic() {
	int n_runtime = 1;
	double train_time = 0, acc;
	FILE* fpout = fopen( "hybrid_multic_batch_topic.txt", "w");
	corpus = new Corpus();
	corpus->dic = corpus->loadDictionary(path+"dic.txt");
	corpus->loadDataGML(path+"20ng_train.gml",
						path+"20ng_test.gml");
	double* test_acc = new double[n_runtime];
	double* test_time = new double[n_runtime];
	HybridMedLDA* sampler[corpus->newsgroupN];
	pthread_t threads[corpus->newsgroupN];
	for( int k_topic = 20; k_topic <= 20; k_topic += 5) {
		for( int ni = 0; ni < n_runtime; ni++) {
			train_time = 0;
			debug( "k_topic = %d, ni = %d\n", k_topic, ni);
			for( int si = 0; si < corpus->newsgroupN; si++) {
				sampler[si] = new HybridMedLDA(corpus, si);
				sampler[si]->K = k_topic;
				pthread_create(&threads[si], NULL, sampler_train, (void*)sampler[si]);
			}
			for(int si = 0; si < corpus->newsgroupN; si++)
				pthread_join(threads[si], NULL);
			acc = 0;
			
			for( int i = 0; i < sampler[0]->testData->D; i++) {
				int label;
				double confidence = 0-INFINITY;
				for( int si = 0; si < corpus->newsgroupN; si++) {
					if( sampler[si]->testData->my[i] > confidence) {
						label = si;
						confidence = sampler[si]->testData->my[i];
					}
				}
				if(sampler[label]->testData->y[i] == 1) {
					acc++;
				}
				printf( "i %d D %d\n", i, sampler[0]->testData->D);
			}
			printf( "D: %d\n", sampler[0]->testData->D);
			fprintf( fpout, "\n");
			test_acc[ni] = (double)acc/(double)sampler[0]->testData->D;
			printf("run[%d]: acc = %lf, D = %d, accuracy = %lf\n", ni, acc, sampler[0]->testData->D, test_acc[ni]);
			test_time[ni] = train_time;
			for( int si = 0; si < corpus->newsgroupN; si++) delete sampler[si];
		}
		double ave_acc = vecsum(test_acc, n_runtime)/n_runtime;
		vecsubs( test_acc,  ave_acc, n_runtime);
		vecmul( test_acc, test_acc, n_runtime);
		double std_acc = sqrt(vecsum(test_acc, n_runtime)/(n_runtime));
		double ave_time = vecsum(test_time, n_runtime)/n_runtime;
		vecsubs( test_time,  ave_time, n_runtime);
		vecmul( test_time, test_time, n_runtime);
		double std_time = sqrt(vecsum(test_time, n_runtime)/(n_runtime));
		fprintf( fpout, "topic %d ave_acc %lf std_acc %lf ave_time %lf std_time %lf\n", k_topic, ave_acc, std_acc, ave_time, std_time);
		fflush(fpout);
	}
	fclose(fpout);
	delete[] test_acc;
	delete[] test_time;
}

void multic_online_topic() {
	int n_runtime = 1;
	double train_time = 0, train_p_time = 0, acc;
	time_t ts_p, te_p; // parallel time.
	clock_t ts, te;    // sequential time.
	FILE* fpout = fopen( "hybrid_multic_online_topic.txt", "w");
	corpus = new Corpus();
	corpus->dic = corpus->loadDictionary(path+"dic.txt");
	corpus->loadDataGML(path+"20ng_train.gml",
						path+"20ng_test.gml");
	double* test_acc = new double[n_runtime];
	double* test_time = new double[n_runtime];
	double* test_p_time = new double[n_runtime];
	HybridMedLDA* sampler[corpus->newsgroupN];
	pthread_t threads[corpus->newsgroupN];
	for( int k_topic = 20; k_topic <= 20; k_topic += 5) {
		for( int ni = 0; ni < n_runtime; ni++) {
			train_time = 0;
			debug("run[%d]: k_topic = %d, ", ni, k_topic);
			for( int si = 0; si < corpus->newsgroupN; si++) {
				sampler[si] = new HybridMedLDA(corpus, si);
				sampler[si]->batchSize = 512;
				sampler[si]->K = k_topic;
				sampler[si]->lets_multic = true;
				sampler[si]->init();
				pthread_create(&threads[si], NULL, sampler_train, (void*)sampler[si]); // train.
			}
			time(&ts_p);
			ts = clock();
			for(int si = 0; si < corpus->newsgroupN; si++)
				pthread_join(threads[si], NULL);
			time(&te_p);
			te = clock();
			train_time = (te-ts)/(double)CLOCKS_PER_SEC;
			train_p_time = difftime(te_p, ts_p);
			debug("training time = %lf, training time parallel = %lf", train_time, train_p_time);
			for( int si = 0; si < corpus->newsgroupN; si++) { // inference.
				pthread_create(&threads[si], NULL, sampler_inference, (void*)sampler[si]);
			}
			for(int si = 0; si < corpus->newsgroupN; si++)
				pthread_join(threads[si], NULL);
			acc = 0;
			
			for( int i = 0; i < sampler[0]->testData->D; i++) {
				int label;
				double confidence = 0-INFINITY;
				for( int si = 0; si < corpus->newsgroupN; si++) {
					if( sampler[si]->testData->my[i] > confidence) {
						label = si;
						confidence = sampler[si]->testData->my[i];
					}
				}
				if(sampler[label]->testData->y[i] == 1) {
					acc++;
				}
			}
			test_acc[ni] = (double)acc/(double)sampler[0]->testData->D;
			debug("accuracy = %lf\n", test_acc[ni]);
			test_time[ni] = train_time;
			test_p_time[ni] = train_p_time;
			for( int si = 0; si < corpus->newsgroupN; si++) delete sampler[si];
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
	delete[] test_acc;
	delete[] test_time;
}

void multic_online_commit() {
	int n_runtime = 3;
	double acc;
	FILE* fpout = fopen( "hybrid_multic_online_commit.txt", "w");
	corpus = new Corpus();
	corpus->dic = corpus->loadDictionary(path+"dic.txt");
	corpus->loadDataGML(path+"20ng_train.gml",
						path+"20ng_test.gml");
	double* test_acc = new double[n_runtime];
	HybridMedLDA* sampler[corpus->newsgroupN];
	pthread_t threads[corpus->newsgroupN];
	for(int bsize = 12; bsize <= 12; bsize += 3) {
		for( int k_topic = 20; k_topic <= 100; k_topic += 5) {
			for( int ni = 0; ni < n_runtime; ni++) {
				for( int si = 0; si < corpus->newsgroupN; si++) {
					sampler[si] = new HybridMedLDA(corpus, si);
					sampler[si]->batchSize = pow(2,bsize);
					sampler[si]->K = k_topic;
					sampler[si]->lets_multic = true;
					sampler[si]->lets_commit = true;
					sampler[si]->commit_point_spacing = corpus->trainDataSize/(5*sampler[si]->batchSize);
					sampler[si]->init();
					pthread_create(&threads[si], NULL, sampler_train, (void*)sampler[si]); // train.
				}
				debug("run[%d]: k_topic = %d, batchsize = %d", ni, k_topic, sampler[0]->batchSize);
				for(int si = 0; si < corpus->newsgroupN; si++)
					pthread_join(threads[si], NULL);
				for( int si = 0; si < corpus->newsgroupN; si++) { // inference.
					pthread_create(&threads[si], NULL, sampler_inference, (void*)sampler[si]);
				}
				for(int si = 0; si < corpus->newsgroupN; si++)
					pthread_join(threads[si], NULL);
				fprintf( fpout, "topic %d batchsize %d J %d ", k_topic, sampler[0]->batchSize, sampler[0]->J);
				for( int ci = 0; ci < sampler[0]->commit_points.size(); ci++) {
					int acc = 0, label = 0;
					for( int i = 0; i < sampler[0]->testData->D; i++) {
						int label;
						double confidence = 0-INFINITY;
						for( int si = 0; si < corpus->newsgroupN; si++) {
							if( sampler[si]->commit_points.at(ci).my[i] > confidence) {
								label = si;
								confidence = sampler[si]->commit_points.at(ci).my[i];
							}
						}
						if(sampler[label]->testData->y[i] == 1) {
							acc++;
						}
					}
					fprintf( fpout, "ob %lf time %lf acc %lf ", sampler[0]->commit_points.at(ci).ob_percent,
							sampler[0]->commit_points.at(ci).time,
							(double)acc/sampler[0]->testData->D);
					for(int si = 0; si < corpus->newsgroupN; si++) {
						delete[] sampler[si]->commit_points.at(ci).my;
					}
				}
				fprintf( fpout, "\n");
				for( int si = 0; si < corpus->newsgroupN; si++) delete sampler[si];
			}
		}
	}
	fclose(fpout);
}

#endif /* defined(__OnlineTopic__experiment__) */
