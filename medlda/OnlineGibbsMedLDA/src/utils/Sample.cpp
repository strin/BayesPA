//
//  Sample.cpp
//  OnlineTopic
//
//  Created by Tianlin Shi on 5/1/13.
//  Copyright (c) 2013 Tianlin Shi. All rights reserved.
//

#include "Sample.h"


SampleZ::SampleZ(int D, stl::vec<int> W) {
	Z = new int*[D];
	Zbar = new double*[D];
	Cdk = NULL;
	memset(Z, 0, sizeof(int*)*D);
	memset(Zbar, 0, sizeof(double*)*D);
	invlambda = new double[D];
	for( int i = 0; i < D; i++) {
		Z[i] = new int[W[i]];
	}
	this->D = D;
	this->W = W;
}


SampleZ::~SampleZ() {
	if(Z != 0) {
		for( int i = 0; i < D; i++)
			if( Z[i] != 0)
				delete[] Z[i];
		delete[] Z;
	}
	if(Zbar != 0) {
		for( int i = 0; i < D; i++)
			if( Zbar[i] != 0)
				delete[] Zbar[i];
		delete[] Zbar;
	}
	if(invlambda != 0)
		delete[] invlambda;
	if(Cdk) {
		for( int i = 0; i < D; i++) {
			delete[] Cdk[i];
		}
		delete[] Cdk;
	}
}
