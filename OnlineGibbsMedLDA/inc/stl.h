#pragma once

#include <vector>
#include <string>
#include <memory>
#include <iostream>

namespace stl {

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

}