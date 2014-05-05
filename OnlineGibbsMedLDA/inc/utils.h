//
//  utils.h
//  OnlineTopic
//
//  Created by Tianlin Shi on 4/30/13.
//  Copyright (c) 2013 Tianlin Shi. All rights reserved.
//

#ifndef __OnlineTopic__utils__
#define __OnlineTopic__utils__


#include <iostream>
#include "dirent.h"
#include "debug.h"
#include "stdlib.h"
#include "time.h"
#include "cokus.h"
#include "math.h"
#include "ap.h"
#include "apaux.h"
#include "spdinverse.h"
#include "time.h"
#include <sys/time.h>
#include <list>
#include <vector>
#include <queue>
#include "objcokus.h"



#define UINT32_MAX ((unsigned int)(-1))

using namespace std;

static string toLowerCase( string str);
static int countFiles( string directory);
/* splitWords
 * split the given text into words and return their lower case.
 */
static list<string>* splitWords( string text);

static double random01() {
	return (double) (((unsigned long) randomMT()) / 4294967296.);
}

static int countFiles( string directory) {
	DIR* dir = opendir( directory.c_str());
	int count = 0;
	struct dirent* entry;
	while( entry = readdir(dir)) {
		if( entry->d_type == DT_DIR || entry->d_name[0] == '.') continue;
		count++;
	}
	return count;
}


static list<string>* splitWords( string text) {
	list<string>* res = new list<string>();
	int ptr = 0;
	string word = "";
	while( ptr < text.length()) {
		if( (text[ptr] >= 'A' && text[ptr] <= 'Z')
		   || (text[ptr] >= 'a' && text[ptr] <= 'z')
		   || (text[ptr] == '-' && !word.empty())) { // is word character.
			word += text[ptr];
		}else{
			if( !word.empty()) {
				res->push_back(toLowerCase(word));
//				debug("%s\n", toLowerCase(word).c_str());
			}
			word = "";
		}
		ptr++;
	}
	return res;
}

static list<int>* splitWords( string text, map<string,int>* dic) {
	list<int>* res = new list<int>();
	int ptr = 0;
	string word = "";
	while( ptr < text.length()) {
		if( (text[ptr] >= 'A' && text[ptr] <= 'Z')
		   || (text[ptr] >= 'a' && text[ptr] <= 'z')
		   || (text[ptr] == '-' && !word.empty())) { // is word character.
			word += text[ptr];
		}else{
			if( !word.empty()) {
				word = toLowerCase(word);
				if( dic->find(word) != dic->end())
					res->push_back( (*dic)[word]);
				//				debug("%s\n", toLowerCase(word).c_str());
			}
			word = "";
		}
		ptr++;
	}
	return res;
}

static string toLowerCase( string str) {
	for( int i = 0; i < str.length(); i++) {
		if( str[i] >= 'A' && str[i] <= 'Z') str[i] = str[i]-'A'+'a';
	}
	return str;
}


/* compute veca = x*veca */
static void vecmuls( double* veca, double x, int spaceK) {
	for( int k = 0; k < spaceK; k++) {
		veca[k] *= x;
	}
}

/* compute veca = veca/x */
static void vecdivs( double* veca, double x, int spaceK) {
	for( int k = 0; k < spaceK; k++) {
		veca[k] /= x;
	}
}

/* compute sum(veca.*vecb) */
static double dotprod( double* veca, double* vecb, int spaceK) {
	double res = 0;
	for( int k = 0; k < spaceK; k++) {
		res += veca[k]*vecb[k];
	}
	return res;
}

/* compute veca = veca+vecb */
static void vecadd( double* veca, double* vecb, int spaceK) {
	for( int k = 0; k < spaceK; k++) {
		veca[k] += vecb[k];
	}
}

/* compute veca = veca-vecb */
static void vecsub( double* veca, double* vecb, int spaceK) {
	for( int k = 0; k < spaceK; k++) {
		veca[k] -= vecb[k];
	}
}

/* compute veca = veca-scalar */
static void vecsubs( double* veca, double scalar, int spaceK) {
	for( int k = 0; k < spaceK; k++) {
		veca[k] -= scalar;
	}
}

/* compute veca = veca-scalar */
static void vecmul( double* veca, double* vecb, int spaceK) {
	for( int k = 0; k < spaceK; k++) {
		veca[k] *= vecb[k];
	}
}

/* compute sum(veca) */
static double vecsum( double* veca, int spaceK) {
	double res = 0;
	for( int k = 0; k < spaceK; k++) {
		res += veca[k];
	}
	return res;
}

/* compute veca.^2 */
static void vecsqr( double* veca, int spaceK) {
	for( int k = 0; k < spaceK; k++)
		veca[k] = veca[k]*veca[k];
}

/* compute vecabs(vecabs) */
static void vecabs( double* veca, int spaceK) {
	for( int k = 0; k < spaceK; k++) {
		veca[k] = fabs(veca[k]);
	}
}

/* free a 2d matrix */
template<class T>
static void free2d( T** m, int n) {
	for( int i = 0; i < n; i++) delete[] m[i];
	delete[] m;
}




static int sampleByImportanceAccum( int* label, double* importance, int n) {
	double sel = importance[n-1]*random01();
	int i;
	for( i = 0; importance[i] < sel; i++);
	if( label)
		return label[i];
	else
		return i;
}


static bool choleskydec(double **A, double **res, const int &n, bool isupper)
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

/* the inverse of a matrix. */
static void inverse_cholydec(double **A, double **res, double **lowerTriangle, const int &n)
{
    ap::real_2d_array a;
    a.setbounds(0, n-1, 0, n-1);
	
	// upper-triangle matrix
	for ( int i=0; i<n; i++ ) {
		for ( int j=0; j<n; j++ ) {
			if ( j < i ) a(i, j) = 0;
			else a(i, j) = A[i][j];
		}
	}
	
    if( spdmatrixcholesky(a, n, true) ) {
		// get cholesky decomposition result.
		double *dPtr = NULL;
		for ( int i=0; i<n; i++ ) {
			dPtr = lowerTriangle[i];
			for ( int j=0; j<=i; j++ ) {
				dPtr[j] = a(j, i);
			}
		}
		
		// inverse
        if( spdmatrixcholeskyinverse(a, n, true) ) {
        } else {
			printf("Inverse matrix error!");
		}
	} else {
		printf("Non-PSD matrix!");
	}
	
	for ( int i=0; i<n; i++ ) {
		for ( int j=0; j<n; j++ ) {
			if ( j < i ) res[i][j] = a(j, i);
			else res[i][j] = a(i, j);
		}
	}
}

template<class type>
static void shuffleArray(type *idx, int D) {
	objcokus cokus;
	cokus.seedMT(time(NULL));
	for(int x = 0; x < D; x++) {
		int y = cokus.randomMT()%D;
		if(x != y) {
			type temp = idx[x];
			idx[x] = idx[y]; idx[y] = temp;
		}
	}
}

// commit the test error of a sampler. testErrorFile should be 0 to create a new file.
static void commitTestError( FILE** testErrorFile, double id, double accuracy) {
	if( *testErrorFile == 0) {
		*testErrorFile = fopen ( "test_error_online_sampler.txt", "w");
	}else{
		*testErrorFile = fopen ( "test_error_online_sampler.txt", "a");
	}
	fprintf( *testErrorFile, "%lf %lf\n", id, accuracy);
	fclose(*testErrorFile);
	*testErrorFile = (FILE*)1;
}


static double digamma(double x) {
	double result = 0, xx, xx2, xx4;
	for ( ; x < 7; ++x)
		result -= 1/x;
	x -= 1.0/2.0;
	xx = 1.0/x;
	xx2 = xx*xx;
	xx4 = xx2*xx2;
	result += log(x)+(1./24.)*xx2-(7.0/960.0)*xx4+(31.0/8064.0)*xx4*xx2-(127.0/30720.0)*xx4*xx4;
	return result;
}



static int clock_gettime(struct timespec* t) {
    struct timeval now;
    int rv = gettimeofday(&now, NULL);
    if (rv) return rv;
    t->tv_sec  = now.tv_sec;
    t->tv_nsec = now.tv_usec * 1000;
    return 0;
}





/* very simple approximation */
static double st_gamma(double x)
{
	return sqrt(2.0*M_PI/x)*pow(x/M_E, x);
}

static double sp_gamma(double z)
{
	const int a = 12;
	static double c_space[12];
	static double *c = NULL;
	int k;
	double accm;
	
	if ( c == NULL ) {
		double k1_factrl = 1.0; /* (k - 1)!*(-1)^k with 0!==1*/
		c = c_space;
		c[0] = sqrt(2.0*M_PI);
		for(k=1; k < a; k++) {
			c[k] = exp(a-k) * pow(a-k, k-0.5) / k1_factrl;
			k1_factrl *= -k;
		}
	}
	accm = c[0];
	for(k=1; k < a; k++) {
		accm += c[k] / ( z + k );
	}
	accm *= exp(-(z+a)) * pow(z+a, z+0.5); /* Gamma(z+1) */
	return accm/z;
}

#endif /* defined(__OnlineTopic__utils__) */
