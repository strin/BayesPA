//
//  debug.h
//  OnlineTopic
//
//  Created by Tianlin Shi on 4/29/13.
//  Copyright (c) 2013 Tianlin Shi. All rights reserved.
//

#ifndef OnlineTopic_debug_h
#define OnlineTopic_debug_h

#include <string.h>
#include "stdarg.h"
#include "stdio.h"
#include <string>
#include <map>

#define DEBUG_PRINT 1

using namespace std;

static int debug_indentation = 0;
static bool debug(char* message, ...) {
	for( int di = 0; di < debug_indentation; di++) printf( "\t");
	if( DEBUG_PRINT) {
		va_list arg;
		va_start(arg, message);
		int done = vprintf(message, arg);
		va_end(arg);
		fflush(stdout);
		return done;
	}
	return 0;
}
static void debug_indent() {
	debug_indentation++;
}
static void debug_unindent() {
	debug_indentation--;
}

#endif
