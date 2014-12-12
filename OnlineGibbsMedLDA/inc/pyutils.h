/*
 utils.h
 OnlineTopic

 Created by Tianlin Shi on 12/07/14.
 Copyright (c) 2013 Tianlin Shi. All rights reserved.
*/
#pragma once

#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/object.hpp>
#include <boost/python/str.hpp>
#include <boost/python/list.hpp>
#include <boost/python/dict.hpp>

#include "stl.h"

namespace pyutils {

using namespace stl;

template<class T>
vec<T> makeVector(boost::python::list array) {
  vec<T> vector;
  for(size_t i = 0; i < boost::python::len(array); i++) {
    vector.push_back(boost::python::extract<T>(array[i]));
  }
  return vector;
}

template<class T>
vec2D<T> makeVector2D(boost::python::list array) {
  vec2D<T> vector2D;
  for(size_t i = 0; i < boost::python::len(array); i++) {
    boost::python::list row = boost::python::extract<boost::python::list>(array[i]);
    std::vector<T> new_row;
    for(size_t j = 0; j < boost::python::len(array[i]); j++) {
      new_row.push_back(boost::python::extract<T>(row[i]));
    }
    vector2D.push_back(new_row);
  }
  return vector2D;
}



}





