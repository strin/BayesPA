//
//  Corpus.cpp
//  OnlineTopic
//
//  Created by Tianlin Shi on 4/29/13.
//  Copyright (c) 2013 Tianlin Shi. All rights reserved.
//
/* A few words on Corpus.
 * * Converts all upper-case to lower case for words.
 */

#include "Corpus.h"
#include "Mapper.h"

Corpus::Corpus() {
	multi_label = false;
}

Corpus::~Corpus() {
	if( documents) { // free document ptr if it is not NULL.
		for( int i = 0; i < newsgroupN; i++) {
			if( !documents[i]) delete documents[i];
		}
		delete documents;
	}
}

bool Corpus::loadRawCorpus( string directory) {
	DIR* dir = opendir( directory.c_str());
	if( dir == 0) {
		debug( "error: directory don't exist when loading corpus\n");
	}
	struct dirent* entry = readdir( dir); // read directory.
	newsgroupN = 0;
	while( entry) {
		if( entry->d_type == DT_DIR && entry->d_name[0] != '.') { // is newsgroup directory.
			debug( "loading corpus group: %s\n", entry->d_name);
			Mapper::newsgroupID[string(entry->d_name)] = newsgroupN;
			Mapper::newsgroupIDInv[newsgroupN] = string(entry->d_name);
			newsgroupN++;
		}
		entry = readdir(dir);
	}
	closedir(dir);
	documents = new Document**[newsgroupN];
	D = new int[newsgroupN];
	int groupi = 0;
	for( map<string,int>::iterator it = Mapper::newsgroupID.begin();
		it != Mapper::newsgroupID.end(); it++, groupi++) {
		string sub_directory = directory+string((*it).first)+"/";
		D[groupi] = countFiles(sub_directory);
		documents[groupi] = new Document*[D[groupi]];
		int filei = 0;
		DIR* subdir = opendir( sub_directory.c_str());
		while( (entry = readdir(subdir))) {
			if( entry->d_type == DT_DIR || entry->d_name[0] == '.') continue;
			((documents[groupi])[filei]) = new Document();
			((documents[groupi])[filei])->name = entry->d_name;
			string file_path = sub_directory+entry->d_name;
			((documents[groupi])[filei])->parseDocument(fopen(file_path.c_str(), "r"));
			((documents[groupi])[filei])->newsgroupId = groupi;
			filei++;
		}
		closedir(subdir);
	}
	return true;
}

map<string,int>* Corpus::genDictionary() {
	map<string,int>* dic = new map<string,int>();
	int wordId = 0;
	for( int groupi = 0; groupi < newsgroupN; groupi++) {
		for( int doci = 0; doci < D[groupi]; doci++) {
			Document* d = (documents[groupi])[doci];
			list<string>* words = splitWords(d->text);
			for( list<string>::iterator it = words->begin(); it != words->end(); it++) {
				if( dic->find((*it)) == dic->end()) {
					(*dic)[(*it)] = wordId++;
				}
			}
			delete words;
		}
	}
	return dic;
}

bool Corpus::saveDictionary(string file_path, map<string,int>* dic) {
	FILE* fpout = fopen( file_path.c_str(), "w");
	for( map<string,int>::iterator it = dic->begin(); it != dic->end(); it++) {
		fprintf( fpout, "%s:%d\n", (*it).first.c_str(), (*it).second);
	}
	fclose(fpout);
	return false;
}

Dictionary* Corpus::loadDictionary( string file_path, bool assign_label) {
	FILE* file = fopen( file_path.c_str(), "r");
	int wid = 0;
	char buf[256];
	Dictionary* res = new Dictionary();
	res->dic = new map<string, int>();
	res->inv_dic = new map<int, string>();
	while( !feof(file)) {
		fgets(buf, 256, file);
		buf[strlen(buf)-1] = '\0';
		(*(res->dic))[string(buf)] = wid;
		(*(res->inv_dic))[wid] = string(buf);
		wid++;
	}
	fclose(file);
	return res;
}

bool Corpus::processRawCorpus() {
	if( dic == 0) {
		debug( "error: no dictionary when processing raw corpus.\n");
		return false;
	}
	int wordId = 0;
	for( int groupi = 0; groupi < newsgroupN; groupi++) {
		for( int doci = 0; doci < D[groupi]; doci++) {
			Document* d = (documents[groupi])[doci];
			list<int>* words = splitWords(d->text, dic->dic);
			d->W = words->size();
			d->words = new int[words->size()];
			int i = 0;
			for( list<int>::iterator it = words->begin(); it != words->end(); it++, i++) {
				d->words[i] = (*it);
			}
			d->inv_dic = this->dic->inv_dic;
			delete words;
		}
	}
	return true;
}

bool Corpus::writeBinaryCorpus(string file_path) {
	FILE* fpout = fopen( file_path.c_str(), "w");
	fprintf( fpout, "%d\n", newsgroupN);
	for( int groupi = 0; groupi < newsgroupN; groupi++) {
		for( int doci = 0; doci < D[groupi]; doci++) {
			Document* d = (documents[groupi])[doci];
			fprintf( fpout, "%d %d %d %s %s\n", groupi, doci, d->W, d->name.c_str(), Mapper::newsgroupIDInv[groupi].c_str());
			for( int i = 0; i < d->W; i++) {
				fprintf( fpout, "%d ", d->words[i]);
			}
			fprintf( fpout, "\n");
		}
	}
	fclose( fpout);
	return true;
}

bool Corpus::loadDataGML(string train_file_path, string test_file_path) {
	FILE* fpin = fopen( train_file_path.c_str(), "r");
	
	fscanf( fpin, "%d", &trainDataSize);
	trainData = new DataSample*[trainDataSize];
	
	newsgroupN = 0;
	T = 0;
	
	int train_pos = 0;
	for( int d = 0; d < trainDataSize; d++) {
		trainData[d] = new DataSample();
		fscanf( fpin, "%d", &trainData[d]->W);
		fscanf( fpin, "%d", &trainData[d]->label);
		if(trainData[d]->label+1 > newsgroupN) newsgroupN = trainData[d]->label+1;
		
		trainData[d]->words = new int[trainData[d]->W];
		for( int i = 0; i < trainData[d]->W; i++) {
			fscanf( fpin, "%d", &trainData[d]->words[i]);
			if(trainData[d]->words[i] > T-1) T = trainData[d]->words[i]+1;
		}
	}
	fclose( fpin);
	
	if(newsgroupN <= 2) {
		for( int d = 0; d < trainDataSize; d++) {
			train_pos += (1+trainData[d]->label)/2;
			trainData[d]->label = trainData[d]->label*2-1; // renormalize to {-1,1}.
		}
	}
	
	fpin = fopen( test_file_path.c_str(), "r");
	fscanf( fpin, "%d", &testDataSize);
	testData = new DataSample*[testDataSize];
	
	int test_pos = 0;
	for( int d = 0; d < testDataSize; d++) {
		testData[d] = new DataSample();
		fscanf( fpin, "%d", &testData[d]->W);
		fscanf( fpin, "%d", &testData[d]->label);
		testData[d]->words = new int[testData[d]->W];
		for( int i = 0; i < testData[d]->W; i++) {
			fscanf( fpin, "%d", &testData[d]->words[i]);
			if(testData[d]->words[i] > T-1) T = testData[d]->words[i]+1;
		}
	}
	fclose( fpin);
	
	if(newsgroupN <= 2) {
		for( int d = 0; d < testDataSize; d++) {
			testData[d]->label = testData[d]->label*2-1; // renormalize to {-1,1}.
			test_pos += (1+testData[d]->label)/2;
		}
		debug( "[corpus] loaded, train split:%d/%d, test split:%d/%d\n", train_pos, trainDataSize-train_pos,
			  test_pos, testDataSize-test_pos);
	}
	
	return true;
}


bool Corpus::loadDataGML_MT(string train_file_path, string test_file_path, bool single_label) {
	multi_label = true;
	int label_n;
	FILE* fpin = fopen( train_file_path.c_str(), "r");
	
	fscanf( fpin, "%d", &trainDataSize);
	trainData = new DataSample*[trainDataSize];
	
	newsgroupN = 0;
	T = 0;
	
	int train_pos = 0;
	for( int d = 0; d < trainDataSize; d++) {
		trainData[d] = new DataSample();
		fscanf( fpin, "%d", &trainData[d]->W);
		int label;
		if(single_label) {
			fscanf(fpin, "%d", &label);
			trainData[d]->multi_label.push_back(label);
			if(label+1 > newsgroupN) newsgroupN = label+1;
		}else{
			fscanf( fpin, "%d", &label_n);
			for(int lbi = 0; lbi < label_n; lbi++) {
				fscanf(fpin, "%d", &label);
				trainData[d]->multi_label.push_back(label);
				if(label+1 > newsgroupN) newsgroupN = label+1;
			}
		}
		trainData[d]->words = new int[trainData[d]->W];
		for( int i = 0; i < trainData[d]->W; i++) {
			fscanf( fpin, "%d", &trainData[d]->words[i]);
			if(trainData[d]->words[i] > T-1) T = trainData[d]->words[i]+1;
		}
	}
	fclose( fpin);
	
	
	fpin = fopen( test_file_path.c_str(), "r");
	fscanf( fpin, "%d", &testDataSize);
	testData = new DataSample*[testDataSize];
	
	int test_pos = 0;
	for( int d = 0; d < testDataSize; d++) {
		testData[d] = new DataSample();
		fscanf( fpin, "%d", &testData[d]->W);
		int label;
		if(single_label) {
			fscanf(fpin, "%d", &label);
			testData[d]->multi_label.push_back(label);
		}else{
			fscanf( fpin, "%d", &label_n);
			for(int lbi = 0; lbi < label_n; lbi++) {
				fscanf(fpin, "%d", &label);
				testData[d]->multi_label.push_back(label);
			}
		}
		testData[d]->words = new int[testData[d]->W];
		for( int i = 0; i < testData[d]->W; i++) {
			fscanf( fpin, "%d", &testData[d]->words[i]);
			if(testData[d]->words[i] > T-1) T = testData[d]->words[i]+1;
		}
	}
	fclose( fpin);
	
	
	return true;
}


bool Corpus::loadBinaryCorpus(string file_path) {
	if( dic == NULL) {
		debug( "error: cannot find dictionary when loading corpus.\n");
		return false;
	}
	FILE* fpin = fopen( file_path.c_str(), "r");
	fscanf( fpin, "%d", &newsgroupN);
	vector<Document*>* v = new vector<Document*>[newsgroupN];
	D = new int[newsgroupN];
	memset( D, 0, sizeof(D));
	
	int groupi, doci, W;
	char name[256];
	char newsgroup[256];
	debug( "> loading newsgroup dataset.\n");
	newsgroupN = 0;
	while( !feof(fpin)) {
		if( fscanf( fpin, "%d %d %d %s %s\n", &groupi, &doci, &W, name, newsgroup) == -1) break;
		if( Mapper::newsgroupIDInv.find(groupi) == Mapper::newsgroupIDInv.end()) {
			Mapper::newsgroupIDInv[groupi] = newsgroup;
			Mapper::newsgroupID[newsgroup] = groupi;
			newsgroupN++;
			debug( "[newsgroup] %d %s\n", groupi, newsgroup);
		}
		D[groupi]++;
		Document* d = new Document();
		d->words = new int[W];
		d->W = W;
		d->inv_dic = dic->inv_dic;
		d->newsgroup = string(newsgroup);
		for( int i = 0; i < W; i++) {
			fscanf( fpin, "%d", &d->words[i]);
		}
		d->newsgroupId = groupi;
		d->name = string(name);
		v[groupi].push_back(d);
	}
	this->documents = new Document**[newsgroupN];
	for( int i = 0; i < newsgroupN; i++) {
		documents[i] = &(v[i][0]);
	}
	fclose( fpin);
	return true;
}

bool Corpus::genTrainingDataRandom( double percent) {
//	srand( (unsigned)time( NULL));
	vector<DataSample*> *trainDataVec = new vector<DataSample*>(),
									*testDataVec = new vector<DataSample*>();
	if( percent < 0 || percent > 1) {
		debug( "error: random percentage is out of range [0,1].\n");
		return false;
	}
	if( documentType != NULL) { // clean document type.
		for( int i = 0; i < newsgroupN; i++) {
			if( documentType[i] != NULL) delete documentType[i];
		}
		delete documentType;
	}
	documentType = new bool*[newsgroupN];
	for( int gi = 0; gi < this->newsgroupN; gi++) {
		documentType[gi] = new bool[D[gi]];
		for( int di = 0; di < D[gi]; di++) {
			double proposed = (double)(randomMT()%1000000)/1000000.0;
			Document* doc = documents[gi][di];
			DataSample* sample = new DataSample();
			sample->W = doc->W;
			sample->label = gi;
			sample->words = doc->words;
			if( proposed <= percent) {// give to training.
				documentType[gi][di] = true;
				trainDataVec->push_back(sample);
			}else{
				documentType[gi][di] = false;
				testDataVec->push_back(sample);
			}
		}
	}
	if( this->trainData != 0) delete[] trainData;
	if( this->testData != 0) delete[] testData;
	this->trainData = &trainDataVec->at(0);
	this->trainDataSize = trainDataVec->size();
	this->testData = &testDataVec->at(0);
	this->testDataSize = testDataVec->size();
	return true;
}

bool Corpus::genTrainingData( int groupid) {
	if( !documentType) {
		debug( "error: document type assigned is NULL. \n");
		return false;
	}
	vector<DataSample*> *trainDataVec = new vector<DataSample*>(),
	*testDataVec = new vector<DataSample*>();
	this->documentType = documentType;
	for( int gi = 0; gi < this->newsgroupN; gi++) {
		for( int di = 0; di < this->D[gi]; di++) {
			Document* doc = documents[gi][di];
			DataSample* sample = new DataSample();
			sample->W = doc->W;
			if( doc->W == 0)
				continue;
			sample->label = (gi == groupid ? 1 : -1);
			sample->words = doc->words;
			if( documentType[gi][di] == true) { // then it is training data.
				trainDataVec->push_back(sample);
			}else{
				testDataVec->push_back(sample);
			}
		}
	}
	this->trainData = &trainDataVec->at(0);
	this->trainDataSize = trainDataVec->size();
	this->testData = &testDataVec->at(0);
	this->testDataSize = testDataVec->size();
	this->documentType = documentType;
	return true;
}

bool Corpus::genBinaryTrainingData( string class1, string class2) {
	if( !documentType) {
		debug( "error: document type assigned is NULL. \n");
		return false;
	}
	vector<DataSample*> *trainDataVec = new vector<DataSample*>(),
	*testDataVec = new vector<DataSample*>();
	this->documentType = documentType;
	int train_pos = 0, train_neg = 0, test_pos = 0, test_neg = 0;
	for( int gi = 0; gi < this->newsgroupN; gi++) {
		for( int di = 0; di < this->D[gi]; di++) {
			Document* doc = documents[gi][di];
			if( doc->newsgroup != class1 && doc->newsgroup != class2) continue;
			DataSample* sample = new DataSample();
			sample->W = doc->W;
			if( doc->W == 0)
				continue;
			sample->label = (doc->newsgroup == class1 ? 1 : -1);
//			if( sample->label == -1 && documentType[gi][di] == 1) {
//				printf( "%d %d\n", gi, di);
//			}
			sample->words = doc->words;
			if( documentType[gi][di] == true) { // then it is training data.
				trainDataVec->push_back(sample);
				train_pos += (sample->label+1)/2;
				train_neg += (1-sample->label)/2;
			}else{
				testDataVec->push_back(sample);
				test_pos += (sample->label+1)/2;
				test_neg += (1-sample->label)/2;
			}
		}
	}
	debug( "Binary Training [%s,%s], train split:%d/%d, test split:%d/%d\n",
			  class1.c_str(), class2.c_str(), train_pos, train_neg, test_pos, test_neg);
	this->trainData = &trainDataVec->at(0);
	this->trainDataSize = trainDataVec->size();
	this->testData = &testDataVec->at(0);
	this->testDataSize = testDataVec->size();
	this->documentType = documentType;
	return true;
}

bool Corpus::genBinaryTrainingData( string class1) {
	if( !documentType) {
		debug( "error: document type assigned is NULL. \n");
		return false;
	}
	vector<DataSample*> *trainDataVec = new vector<DataSample*>(),
	*testDataVec = new vector<DataSample*>();
	this->documentType = documentType;
	// count documents to balance posi/neg examples.
	int doc1 = 0, docp = 0;
	for( int gi = 0; gi < this->newsgroupN; gi++) {
		for( int di = 0; di < this->D[gi]; di++) {
			Document* doc = documents[gi][di];
			if( doc->newsgroup == class1) {
				doc1++;
			}
			docp++;
		}
	}
	int train_pos = 0, train_neg = 0, test_pos = 0, test_neg = 0;
	for( int gi = 0; gi < this->newsgroupN; gi++) {
		for( int di = 0; di < this->D[gi]; di++) {
			Document* doc = documents[gi][di];
			if( doc->newsgroup != class1
			   && (double)randomMT()/(double)UINT32_MAX >= (double)doc1/(double)(docp-doc1)) // join or not, flip a coin.
				continue;
			DataSample* sample = new DataSample();
			sample->W = doc->W;
			if( doc->W == 0)
				continue;
			sample->label = (doc->newsgroup == class1 ? 1 : -1);
			sample->words = doc->words;
			if( documentType[gi][di] == true) { // then it is training data.
				trainDataVec->push_back(sample);
				train_pos += (sample->label+1)/2;
				train_neg += (1-sample->label)/2;
			}else{
				testDataVec->push_back(sample);
				test_pos += (sample->label+1)/2;
				test_neg += (1-sample->label)/2;
			}
		}
	}
	debug( "Binary Training [%s], train split:%d/%d, test split:%d/%d\n",
		  class1.c_str(), train_pos, train_neg, test_pos, test_neg);
	this->trainData = &trainDataVec->at(0);
	this->trainDataSize = trainDataVec->size();
	this->testData = &testDataVec->at(0);
	this->testDataSize = testDataVec->size();
	this->documentType = documentType;
	return true;
}

bool Corpus::genMulticlassTrainingData() {
	if( !documentType) {
		debug( "error: document type assigned is NULL. \n");
		return false;
	}
	vector<DataSample*> *trainDataVec = new vector<DataSample*>(),
	*testDataVec = new vector<DataSample*>();
	this->documentType = documentType;
	for( int gi = 0; gi < this->newsgroupN; gi++) {
		for( int di = 0; di < 10; di++) {
			Document* doc = documents[gi][di];
			DataSample* sample = new DataSample();
			sample->W = doc->W;
			if( doc->W == 0)
				continue;
			sample->label = doc->newsgroupId;
			sample->words = doc->words;
			if( documentType[gi][di] == true) { // then it is training data.
				trainDataVec->push_back(sample);
			}else{
				testDataVec->push_back(sample);
			}
		}
	}
	this->trainData = &trainDataVec->at(0);
	this->trainDataSize = trainDataVec->size();
	this->testData = &testDataVec->at(0);
	this->testDataSize = testDataVec->size();
	this->documentType = documentType;
	debug( "Multiclass Training, train: %d/%d, test:%d/%d\n",
		  this->trainDataSize, this->testDataSize);
	return true;
}

bool Corpus::loadDocumentType( char* filename) {
	FILE* fpin = fopen( filename, "r");
	if( documentType != NULL) { // clean document type.
		for( int i = 0; i < newsgroupN; i++) {
			if( documentType[i] != NULL) delete documentType[i];
		}
		delete documentType;
	}
	if( !fpin) {
		debug( "error: document type file not found.\n");
		return false;
	}
	int newsgroupN;
	fscanf( fpin, "%d", &newsgroupN);
	if( newsgroupN != this->newsgroupN) {
		debug( "error: newsgroup number inconsistent.\n");
		return false;
	}
	documentType = new bool*[newsgroupN];
	for( int gi = 0; gi < this->newsgroupN; gi++) {
		int dgi;
		fscanf( fpin, "%d\n", &dgi);
		if( dgi != D[gi]) {
			debug( "error: document number inconsisitent. \n");
			return false;
		}
		documentType[gi] = new bool[D[gi]];
		for( int di = 0; di < this->D[gi]; di++) {
			fscanf( fpin, "%d", &documentType[gi][di]);
		}
	}
}
bool Corpus::saveDocumentType( char* filename) {
	FILE* fpout = fopen( filename, "w");
	if( !fpout) {
		debug( "error: document type file not found.\n");
		return false;
	}
	if( !documentType) {
		debug( "error: document type is NULL.\n");
		return false;
	}
	fprintf( fpout, "%d\n", newsgroupN);
	for( int gi = 0; gi < this->newsgroupN; gi++) {
		fprintf( fpout, "%d\n", D[gi]);
		for( int di = 0; di < this->D[gi]; di++) {
			fprintf( fpout, "%d ", documentType[gi][di]);
		}
		fprintf( fpout, "\n");
	}
	fclose(fpout); // be sure to close fpout, since it will be open soon.
	return true;
}

CorpusData* Corpus::exportTestData( int category) {
	CorpusData* testData = new CorpusData();
	Corpus* corpus = this;
	
	testData->D							= (int)corpus->testDataSize;
	testData->W							= new int[testData->D];
	testData->data						= new int*[testData->D];
	testData->y							= new int[testData->D];
	testData->py						= new int[testData->D];
	testData->my						= new double[testData->D];

	for( int i = 0; i < testData->D; i++) {
		testData->W[i] = corpus->testData[i]->W;
		if( category == -1)
			testData->y[i] = corpus->testData[i]->label;
		else
			testData->y[i] = corpus->testData[i]->label == category ? 1 : -1;
		testData->py[i] = 0;
		testData->data[i] = new int[testData->W[i]];
		//		printf( "[%d] %d\n", i, testData->y[i]);
		for( int w = 0; w < testData->W[i]; w++) testData->data[i][w] = corpus->testData[i]->words[w];
	}
	return testData;
}

