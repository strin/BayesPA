//
//  Sample.h
//  OnlineTopic
//
//  Created by Tianlin Shi on 5/1/13.
//  Copyright (c) 2013 Tianlin Shi. All rights reserved.
//
/*
 * Sample Representation of Document Corpus.
 */

#ifndef __OnlineTopic__Sample__
#define __OnlineTopic__Sample__

#include <iostream>
#include "utils.h"
#include "stl.h"

class SampleZ {
public:
	// construction & destruction.
	SampleZ( int D, stl::vec<int> W);
	~SampleZ();
	// parameters.
	int D;
        stl::vec<int> W;
	// content.
	int** Z; // samples of hidden units.
	double** Cdk;
	double** Zbar;
	double* invlambda; // data augmentation.
};
#endif /* defined(__OnlineTopic__Sample__) */
