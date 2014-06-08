//
//  Mapper.h
//  OnlineTopic
//
//  Created by Tianlin Shi on 4/30/13.
//  Copyright (c) 2013 Tianlin Shi. All rights reserved.
//

#ifndef __OnlineTopic__Mapper__
#define __OnlineTopic__Mapper__

#include <iostream>
#include "debug.h"

class Mapper {
public:
	static std::map<std::string, int> newsgroupID;
	static std::map<int, std::string> newsgroupIDInv;
};
#endif /* defined(__OnlineTopic__Mapper__) */
