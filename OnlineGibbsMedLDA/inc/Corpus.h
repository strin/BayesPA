#ifndef __OnlineTopic__Corpus__
#define __OnlineTopic__Corpus__

#include <iostream>
#include <string>
#include <dirent.h>
#include "debug.h"
#include "utils.h"


using namespace std;

typedef struct {
	double *y;					// label and predicted label (allow multi-task).
	int *words;					// words in the document.
	int nd;						// number of words in the document.
	int label_n;				// number of labels.
}Document;

typedef struct{
	int D;						// total number of documents.
	Document* doc; 				// documents.
}CorpusData;


class Corpus {
public:
	Corpus();
	
	/* load .gml data */
	bool loadDataGML( string train_file_path, string test_file_path, bool multi_task = false, bool raw = false); 

	/* basic parameters */
	int newsgroup_n; 				// total number of newsgroups.
	int m_T;  						// total number of different words.
	bool multi_label;				// is this a multi-label corpus.
	
	/* data */
	CorpusData train_data, test_data;
	map<string, int> word_map, tag_map;
private:
	bool loadDataDocument(FILE *fpin, Document &doc);
	bool loadDataDocumentRaw(FILE* fpin, Document &doc);
};

#endif /* defined(__OnlineTopic__Corpus__) */
