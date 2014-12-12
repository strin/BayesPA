//
//  Setting.cpp
//  OnlineTopic
//
//  Created by Tianlin Shi on 4/29/13.
//  Copyright (c) 2013 Tianlin Shi. All rights reserved.
//

#include "Setting.h"

Setting::Setting() {
	this->m = new map<string, string>();
}

Setting::~Setting() {
	if( !this->m) delete this->m;
}

bool Setting::loadSetting( string file_path) {
	FILE* file = fopen( file_path.c_str(), "r");
	if( !file) {
		debug( "error: setting file not exist.");
		return false;
	}
	while( !feof( file)) {
		string key = readNext( file);
		if( key.empty()) break;
		string value = readNext( file);
		(*m)[key] = value;
	}
	return true;
	fclose(file);
}

string& Setting::operator[]( const string& key) {
	return (*m)[key];
}

string Setting::readNext( FILE* file) {
	string res;
	char ch;
	while( !feof( file)) {
		if( fscanf( file, "%c", &ch) == -1) break;
		if( ch != '\n' && ch != '\r' && ch != '=') {
			res += ch;
		}else{
			break;
		}
	}
	return res;
}


