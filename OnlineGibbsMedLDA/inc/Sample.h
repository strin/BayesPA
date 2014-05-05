#ifndef __OnlineTopic__Sample__
#define __OnlineTopic__Sample__

#include <iostream>
#include "utils.h"
#include "Corpus.h"

class Sample {
public:
	// construction & destruction.
	Sample( int K, int T);
	~Sample();
	// parameters.
	int K, T;
	// content.
	double** phi; // dictionary.
	double* eta;
};

class SampleZ {
public:
	// construction & destruction.
	SampleZ(CorpusData *data);
	~SampleZ();
	// parameters.
	int D;
	// content.
	int** Z; // samples of hidden units.
	double** Cdk;
	double** Zbar;
	double* invlambda; // data augmentation.
};
#endif /* defined(__OnlineTopic__Sample__) */
