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

#include "Document.h"

using namespace std;

typedef struct {
	map<string, int>* dic;
	map<int, string>* inv_dic;
}Dictionary;

typedef struct{
	int* words;
	int W;
	int label;
	vector<int> multi_label;
}DataSample;

typedef struct{
	int D;
	int** data;
	int* y, *py; // label and predicted label.
	double* my; // confidence level (margin).
	int *W;
}CorpusData;

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
