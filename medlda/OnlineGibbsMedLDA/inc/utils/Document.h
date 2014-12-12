//
//  Document.h
//  OnlineTopic
//
//  Created by Tianlin Shi on 4/30/13.
//  Copyright (c) 2013 Tianlin Shi. All rights reserved.
//

#ifndef __OnlineTopic__Document__
#define __OnlineTopic__Document__

#include <iostream>
#include <string>
#include <map>

using namespace std;

class Document {
public:
	// basic functions.
	bool parseDocument( FILE* file);
	void visualize( FILE* fpout, bool text); // visualize as a corpus.
	
	// basic properties.
	string name; // name of the document.
	string newsgroup; // newsgroup it belongs to.
	int newsgroupId; // newsgroup mapped to newsgroupID by Mapper.
	int W; // length of the document.
	int* words; // the set of words in this document.
	string text; // text of the document.
	
	map<int, string>* inv_dic;
private:
	string readKey( FILE* file);
	string readValue( FILE* file);
	string readNextAll( FILE* file);
};

#endif /* defined(__OnlineTopic__Document__) */
