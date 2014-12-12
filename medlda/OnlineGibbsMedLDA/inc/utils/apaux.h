//
//  apaux.h
//  OnlineTopic
//
//  Created by Tianlin Shi on 5/3/13.
//  Copyright (c) 2013 Tianlin Shi. All rights reserved.
// Utils for AP Library.

#ifndef OnlineTopic_apaux_h
#define OnlineTopic_apaux_h

#include "ap.h"
#include "debug.h"

namespace ap {
	static real_2d_array* eye( int n) {
		real_2d_array* res_p = new ap::real_2d_array();
		real_2d_array& res = (*res_p);
		res.setbounds( 0, n-1, 0, n-1);
		for( int i = 0; i < n; i++)
			for( int j = 0; j < n; j++)
				res(i,j) = 0;
		for( int i = 0; i < n; i++)
			res(i,i) = 1;
		return res_p;
	}
	
	static real_2d_array* zeros( int m, int n) {
		real_2d_array* res_p = new ap::real_2d_array();
		real_2d_array& res = (*res_p);
		res.setbounds( 0, m-1, 0, n-1);
		for( int i = 0; i < m; i++)
			for( int j = 0; j < n; j++)
				res(i,j) = 0;
		return res_p;
	}
	
	static void visualize( real_2d_array& array) {
		for( int i = array.getlowbound(1); i <= array.gethighbound(1); i++) {
			for( int j = array.getlowbound(2); j <= array.gethighbound(2); j++) {
				debug( "%lf ", array(i,j));
			}
			debug("\n");
		}
	}
}
#endif
