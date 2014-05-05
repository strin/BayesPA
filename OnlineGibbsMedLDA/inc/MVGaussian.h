//
//  MVGaussian.h
//  OnlineTopic
//
//  Created by Tianlin Shi on 5/3/13.
//  Copyright (c) 2013 Tianlin Shi. All rights reserved.
//

#ifndef __OnlineTopic__MVGaussian__
#define __OnlineTopic__MVGaussian__

#include <iostream>
#include "ap.h"
#include "objcokus.h"

class MVGaussian
{
public:
	MVGaussian(void);
	~MVGaussian(void);
	
	void nextMVGaussian(double *mean, double **precision, double *res, const int &n);
	void nextMVGaussianWithCholeskyAp(ap::real_2d_array& mean, ap::real_2d_array& precisionLowerTriangular, ap::real_2d_array& res);
	void nextMVGaussianWithCholesky(double *mean, double **precisionLowerTriangular, double *res, const int &n);

	double nextGaussian();
	bool choleskydec(double **A, double **res, const int &n, bool isupper);
	
private:
	// for Gaussian random variable
	int m_iSet;
	double m_dGset;
	objcokus cokus;
};

#endif /* defined(__OnlineTopic__MVGaussian__) */
