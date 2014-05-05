//
//  MVGaussian.cpp
//  OnlineTopic
//
//  Created by Tianlin Shi on 5/3/13.
//  Copyright (c) 2013 Tianlin Shi. All rights reserved.
//

#include "MVGaussian.h"
#include "defs.h"
#include "cokus.h"
#include "cholesky.h"

MVGaussian::MVGaussian(void)
{
	m_iSet = 0;
	cokus.reloadMT();
	cokus.seedMT(time(NULL));
}

MVGaussian::~MVGaussian(void)
{
}

void MVGaussian::nextMVGaussian(double *mean, double **precision, double *res, const int &n)
{
	double **precisionLowerTriangular = new double*[n];
	for ( int i=0; i<n; i++ ) {
		precisionLowerTriangular[i] = new double[n];
	}
	choleskydec(precision, precisionLowerTriangular, n, false);
	
//	return nextMVGaussianWithCholesky(mean, precisionLowerTriangular, res, n);
}

void MVGaussian::nextMVGaussianWithCholeskyAp(ap::real_2d_array& mean, ap::real_2d_array& precisionLowerTriangular, ap::real_2d_array& res) {
	// test validity of the shape.
	ap::ap_error::make_assertion( mean.getlowbound(1) == 0 && mean.getlowbound(2) == 0
								 && precisionLowerTriangular.getlowbound(1) == 0
								 && precisionLowerTriangular.getlowbound(2) == 0
								 && res.getlowbound(1) == 0 && res.getlowbound(2) == 0);
	ap::ap_error::make_assertion( mean.gethighbound(1) == precisionLowerTriangular.gethighbound(1)
								 && precisionLowerTriangular.gethighbound(1) == precisionLowerTriangular.gethighbound(2));
	
	// generate sample from unit gaussian.
	ap::real_2d_array sample;
	sample.setshape(mean.shape(1), mean.shape(2));
	for( int i = 0; i <= sample.gethighbound(1); i++) {
		sample(i,0) = this->nextGaussian();
	}
	res = sample;
	
	// back substitution.
	double innerProduct = 0;
	for (int i = sample.shape(1)-1; i >= 0; i--) {
		innerProduct = 0;
		for (int j = i+1; j < sample.shape(1); j++) {
			// the cholesky decomp got us the precisionLowerTriangular triangular
			//  matrix, but we really want the transpose.
			innerProduct += sample(j,0) * precisionLowerTriangular(j,i);
		}
		res(i,0) = (res(i,0) - innerProduct) / precisionLowerTriangular(i,i);
	}
}

void MVGaussian::nextMVGaussianWithCholesky(double *mean, double **precisionLowerTriangular, double *res, const int &n)
{
	// Initialize vector z to standard normals
	//  [NB: using the same array for z and x]
	for (int i = 0; i < n; i++) {
		res[i] = nextGaussian();
	}
	
	// Now solve trans(L) x = z using back substitution
	double innerProduct;
	
	for (int i = n-1; i >= 0; i--) {
		innerProduct = 0;
		for (int j = i+1; j < n; j++) {
			// the cholesky decomp got us the precisionLowerTriangular triangular
			//  matrix, but we really want the transpose.
			innerProduct += res[j] * precisionLowerTriangular[j][i];
		}
		
		res[i] = (res[i] - innerProduct) / precisionLowerTriangular[i][i];
	}
	
	for (int i = 0; i < n; i++) {
		res[i] += mean[i];
	}
}

double MVGaussian::nextGaussian()
{
	if ( m_iSet == 0 ) {
		double dRsq = 0;
		double v1, v2;
		do {
			v1 = 2.0 * cokus.random01() - 1.0;
			v2 = 2.0 * cokus.random01() - 1.0;
			dRsq = v1 * v1 + v2 * v2;
		} while (dRsq > 1.0 || dRsq < 1e-300);
		
		double dFac = sqrt(-2.0 * log(dRsq) / dRsq);
		m_dGset = v1 * dFac;
		m_iSet = 1;
		return v2 * dFac;
	} else {
		m_iSet = 0;
		return m_dGset;
	}
}
bool MVGaussian::choleskydec(double **A, double **res, const int &n, bool isupper)
{
    ap::real_2d_array a;
    a.setbounds(0, n-1, 0, n-1);
	
	if ( isupper ) {
		// upper-triangle matrix
		for ( int i=0; i<n; i++ ) {
			for ( int j=0; j<n; j++ ) {
				if ( j < i ) a(i, j) = 0;
				else a(i, j) = A[i][j];
			}
		}
	} else {
		// lower-triangle matrix
		for ( int i=0; i<n; i++ ) {
			for ( int j=0; j<n; j++ ) {
				if ( j < i ) a(i, j) = A[i][j];
				else a(i, j) = 0;
			}
		}
	}
	
	//printf("\n\n");
	//printmatrix(A, n);
	
	bool bRes = true;
	if ( !spdmatrixcholesky(a, n, isupper) ) {
		printf("matrix is not positive-definite\n");
		bRes = false;
	}
	
	if ( isupper ) {
		for ( int i=0; i<n; i++ ) {
			for ( int j=0; j<n; j++ ) {
				if ( j < i ) res[i][j] = 0;
				else res[i][j] = a(i, j);
			}
		}
	} else {
		for ( int i=0; i<n; i++ ) {
			for ( int j=0; j<n; j++ ) {
				if ( j < i ) res[i][j] = a(i, j);
				else res[i][j] = 0;
			}
		}
	}
	//printf("\n\n");
	//printmatrix(res, n);
	
	return bRes;
}

