#pragma once

#include <vector>
#include <string>
#include <memory>
#include <iostream>

namespace stl {

template<class T>
using ptr = std::shared_ptr<T>;

template<typename T>
class vec {
public:
  vec() {
    this->data = std::make_shared<std::vector<T> >();
  }
  
  vec(size_t size) {
    this->data = std::make_shared<std::vector<T> >(size);
  }

  vec(size_t size, T val) {
    this->data = std::make_shared<std::vector<T> >(size, val);
  }

  T& operator[](size_t index) {
    return (*this->data)[index];
  }

  const T& operator[](size_t index) const {
    return (*this->data)[index];
  }

  vec<T>& operator=(const vector<T>& v) {
    *this->data = v;
    return *this;
  }

  std::vector<T>& operator*() {
    return *this->data;
  }

  void resize(size_t size, T val = 0) {
    this->data->resize(size, val);
  }

  size_t size() const {
    return this->data->size();
  }

  void clear() {
    this->data->clear();
  }
  
  void push_back(const T& val) {
    this->data->push_back(val);
  }

  typedef typename std::vector<T>::iterator IteratorType;
  typedef typename std::vector<T>::const_iterator ConstIteratorType;

  IteratorType begin() {
    return this->data->begin();
  }

  IteratorType end() {
    return this->data->end();
  }

  ConstIteratorType begin() const {
    return this->data->begin();
  }

  ConstIteratorType end() const {
    return this->data->end();
  }

private:
  ptr<std::vector<T> > data;
};

template<typename T>
class vec2D {
public:
  vec2D() {
    this->data = std::make_shared<std::vector<std::vector<T> > >();
  }

  vec2D(size_t size) {
    this->data = std::make_shared<std::vector<std::vector<T> > >(size);
  }

  vec2D(size_t size, const std::vector<T>& val) {
    this->data = std::make_shared<std::vector<std::vector<T> > >(size, val);
  }
  
  void resize(size_t size) {
    this->data->resize(size);
  }

  void resize(size_t size, const std::vector<T>& val) {
    this->data->resize(size, val);
  }

  std::vector<T>& operator[](size_t index) {
    return (*this->data)[index];
  }

  const std::vector<T>& operator[](size_t index) const {
    return (*this->data)[index];
  }

  std::vector<std::vector<T> >& operator*() {
    return *this->data;
  }

  size_t size() const {
    return this->data->size();
  }

  void clear() {
    this->data->clear();
  }

  void push_back(const std::vector<T>& val) {
    this->data->push_back(val);
  }

  void push_back(const vec<T>& val) {
    this->data->push_back(*val.data);
  }
  
  typedef typename std::vector<std::vector<T> >::iterator IteratorType;
  typedef typename std::vector<std::vector<T> >::const_iterator ConstIteratorType;

  IteratorType begin() {
    return this->data->begin();
  }

  IteratorType end() {
    return this->data->end();
  }

  ConstIteratorType begin() const {
    return this->data->begin();
  }

  ConstIteratorType end() const {
    return this->data->end();
  }

private:
  ptr<std::vector<std::vector<T> > > data;
};


}
