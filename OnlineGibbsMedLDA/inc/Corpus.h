//
//  Corpus.h
//  OnlineTopic
//
//  Created by Tianlin Shi on 4/29/13.
//  Copyright (c) 2013 Tianlin Shi. All rights reserved.
//

#ifndef __OnlineTopic__Corpus__
#define __OnlineTopic__Corpus__

#include <iostream>
#include <string>
#include <dirent.h>
#include "debug.h"
#include "utils.h"
#include "stl.h"

#include "Document.h"

using namespace std;
using namespace stl;

typedef struct {
	map<string, int>* dic;
	map<int, string>* inv_dic;
}Dictionary;

class DataSample {
public:
	DataSample() {}
	/* create data sample for multi-class classification */
	DataSample(const vector<int>& ex, const int& label) {
		this->W = ex.size();
		if(this->W == 0) return;
		this->words = new int[this->W];
		this->label = label;
	}
	int* words;
	int W;
	int label;
	vector<int> multi_label;
};

class CorpusData {
public:
	CorpusData() {}
	/* make a copy of data sample, relabel the labels in data according to <category> 
	 * if y == cat, then doc is labelled as 1; 
	 * else, doc is labelled as 0.
	 */
	CorpusData(DataSample** data, int data_size, int category, bool multi_label = false) {
		this->D = data_size;
		this->W	= new int[this->D];
		this->data = new int*[this->D];
		this->y	= new int[this->D];
		this->py = new int[this->D];
		this->my = new double[this->D];

		this->train_pos = this->train_neg = 0;

		for(int i = 0; i < this->D; i++) {
			this->W[i] = data[i]->W;
			if(category == -1)
				this->y[i] = data[i]->label;
			else{
				if(multi_label) {
					this->y[i] = -1;
					for(int j = 0; j < data[i]->multi_label.size(); j++) {
						if(category == data[i]->multi_label[j])
							this->y[i] = 1;
					}
				}else
					this->y[i] = data[i]->label == category ? 1 : -1;
			}
			this->train_neg += (1-this->y[i])/2;
			this->train_pos += (1+this->y[i])/2;
			this->py[i] = 0;
			this->data[i] = new int[this->W[i]];
			for(int w = 0; w < this->W[i]; w++) this->data[i][w] = data[i]->words[w];
		}
	}

	/* destructors */
	~CorpusData() {
		delete[] this->W;
		delete[] this->y;
		delete[] this->py;
		delete[] this->my;
		for(int i = 0; i < this->D; i++) {
			delete[] this->data[i];
		}
		delete[] this->data;
	}

	int D;
	int** data;
	int* y, *py; // label and predicted label.
	double* my; // confidence level (margin).
	int *W;

	int train_pos, train_neg;
};

typedef struct {
	int D;
	int** data;
	int** y, **py;
	double** my;
	int* W;
}CorpusDataMt;

class Corpus {
public:
	Corpus();
	~Corpus();
	
	// Corpus IO.
	bool loadRawCorpus( string directory);
	bool processRawCorpus(); // process the raw corpus.
	bool writeBinaryCorpus( string file_path);
	bool loadBinaryCorpus( string file_path);
	bool loadDataGML( string train_file_path,
								string test_file_path); // load .gml data file.
	bool loadDataGML_MT(string train_file_path, string test_file_path, bool single_label = false);
	// Dictionary IO.
	map<string, int>* genDictionary();
	bool saveDictionary( string file_path, map<string, int>* dic);
	Dictionary* loadDictionary( string file_path, bool assign_label = false);
	
	// Basic Learning Aux.
	// Generate Training Data (do not copy on right).
	bool genTrainingDataRandom( double percent); // generate training/test data at random.
	bool genTrainingData( int groupid); // generate training/test data from docType.
	bool genBinaryTrainingData( string class1, string class2); // generate binary training/test data.
	bool genBinaryTrainingData( string class1);
	bool genMulticlassTrainingData();
	bool loadDocumentType( char* filename);
	bool saveDocumentType( char* filename);
	CorpusData* exportTestData( int category = -1); // export test data in the format of CorpusData*.
	
	// Basic Data.
	Document*** documents;
	int newsgroupN; // total news groups.
	int* D, T; // D: document count for each group, T: total word count.
	Dictionary* dic;
	
	// Training Data.
	bool** documentType; // is it a training document.
	DataSample **trainData, **testData;
	bool multi_label;
	int trainDataSize, testDataSize;
};

#endif /* defined(__OnlineTopic__Corpus__) */
