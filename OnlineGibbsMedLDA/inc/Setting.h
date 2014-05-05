//
//  Setting.h
//  OnlineTopic
//
//  Created by Tianlin Shi on 4/29/13.
//  Copyright (c) 2013 Tianlin Shi. All rights reserved.
//

#ifndef __OnlineTopic__Setting__
#define __OnlineTopic__Setting__

#include <iostream>

#include "stdio.h"
#include "debug.h"
#include <string>
#include <map>

using namespace std;

class Setting {
public:
	Setting();
	~Setting();
	bool loadSetting( string file_path); // load setting from a text file.
	string& operator[]( const string& key); // load/store value at given key.
private:
	string readNext( FILE* file);
	map<string,string>* m; // stores the keyed map.
};

#endif /* defined(__OnlineTopic__Setting__) */
