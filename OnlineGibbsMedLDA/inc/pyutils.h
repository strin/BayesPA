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

#include <vector>
#include <string>
#include <memory>
#include <iostream>


namespace pyutils {

namespace bp = boost::python;

template<class T>
using ptr = std::shared_ptr<T>;

template<class T>
using vec = ptr<std::vector<T> >;

template<class T>
using vec2D = ptr<std::vector<std::vector<T> > >;

template<class T>
vec<T> makeVector() {
  return std::make_shared<std::vector<T> >();
}

template<class T>
vec2D<T> makeVector2D() {
  return std::make_shared<std::vector<std::vector<T> > >();
}

template<class T>
vec<T> makeVector(bp::list array) {
  auto vector = makeVector<T>();
  for(size_t i = 0; i < bp::len(array); i++) {
    vector->push_back(bp::extract<T>(array[i]));
  }
  return vector;
}

template<class T>
vec2D<T> makeVector2D(bp::list array) {
  auto vector2D = makeVector2D<T>();
  for(size_t i = 0; i < bp::len(array); i++) {
    bp::list row = bp::extract<bp::list>(array[i]);
    std::vector<T> new_row;
    for(size_t j = 0; j < bp::len(array[i]); j++) {
      new_row.push_back(bp::extract<T>(row[i]));
    }
    vector2D->push_back(new_row);
  }
  return vector2D;
}



}





