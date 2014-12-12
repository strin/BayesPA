	//
//  Document.cpp
//  OnlineTopic
//
//  Created by Tianlin Shi on 4/30/13.
//  Copyright (c) 2013 Tianlin Shi. All rights reserved.
//

#include "Document.h"
#include "debug.h"
#include "Mapper.h"

bool Document::parseDocument( FILE* file) {
	if( !file) {
		debug( "error: file doesn't exist when parsing document\n");
		return false;
	}
	while( !feof(file)) {
		string key = readKey(file);
		if( key.empty()) break;
		if( key == "\n") {
			text = readNextAll(file);
			break;
		}
		string value = readValue(file);
		if( key == "Newsgroups") {
			newsgroup = value;
			if( Mapper::newsgroupID.empty()
			   || Mapper::newsgroupID.find(newsgroup) != Mapper::newsgroupID.end()) {
//				debug( "error: invalid newsgroup name\n");
			}
			newsgroupId = Mapper::newsgroupID[newsgroup];
		}
	}
	fclose(file);
	return true;
}

string Document::readKey( FILE* file) {
	string res;
	char ch;
	while( !feof( file)) {
		if( fscanf( file, "%c", &ch) == -1) break;
		if( ch == '\n' && res.empty()) return "\n";
		if( ch != '\n' && ch != '\r' && ch != ':') {
			res += ch;
		}else{
			break;
		}
	}
	return res;
}

string Document::readValue( FILE* file) {
	string res;
	char ch;
	while( !feof( file)) {
		if( fscanf( file, "%c", &ch) == -1) break;
		if( ch != '\n' && ch != '\r') {
			res += ch;
		}else{
			break;
		}
	}
	return res;
}


string Document::readNextAll( FILE* file) {
	string res;
	char ch;
	while( !feof( file)) {
		if( fscanf( file, "%c", &ch) == -1) break;
		res += ch;
	}
	return res;
}

void Document::visualize( FILE* fpout, bool text) {
	if( inv_dic == 0) {
		debug( "error: failed to visualize the corpus because dictionary not found.\n");
	}
	if(text) {
		for( int i = 0; i < W; i++) {
			fprintf( fpout, "%s ", (*inv_dic)[words[i]].c_str());
		}
		fprintf( fpout, "\n");
	}else{
		for( int i = 0; i < W; i++) {
			fprintf( fpout, "%d ", words[i]);
		}
		fprintf( fpout, "\n");
	}
}