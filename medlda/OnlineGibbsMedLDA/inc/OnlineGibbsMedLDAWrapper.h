#ifndef PY_PAMEDLDA_GIBBS_WRAPPER_H
#define PY_PAMEDLDA_GIBBS_WRAPPER_H

#include <iostream>
#include <vector>
#include <thread>
#include <cmath>
#include <algorithm>

#include "debug.h" 
#include "pyutils.h"
#include "OnlineGibbsMedLDA.h"

using std::shared_ptr;
using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::thread;

namespace bp = boost::python;

struct sortable {
    double value;
    int index;
    bool operator < (const struct sortable& x) const {
        return value < x.value;
    }
};

class paMedLDAgibbsWrapper {
public:
	paMedLDAgibbsWrapper(boost::python::dict config);
	~paMedLDAgibbsWrapper();

	void train(bp::list batch, bp::list label);
	bp::list infer(bp::list batch, bp::list label, bp::object num_test_sample, bool point_estimate);
	bp::object timeElapsed() const;
	bp::object testAcc() const {return bp::object(m_test_acc); }
	bp::list topicMatrix(bp::object category_no) const;
	bp::list topWords(bp::object category_no, int topk) const;
	bp::list topicDistOfInference(bp::object category_no) const;

	vector<shared_ptr<OnlineGibbsMedLDA> > pamedlda;

	bp::object numWord() const;
	bp::object numLabel() const;

	double m_test_acc;

private:
  /* return number of words in the vocabulary */
  size_t _numWord;
  size_t _numLabel;

  static pair<vec2D<int>, vec<int> > filterWordAndLabel(bp::list batch, bp::list label, size_t T, size_t C) {
    auto ret = vec2D<int>();
    vec<int> new_label;
    for(size_t ni = 0; ni < bp::len(batch); ni++) {
      bp::list ex = bp::extract<bp::list>(batch[ni]);
      size_t y = (size_t)bp::extract<int>(label[ni]);
      if(y >= C) continue;
      std::vector<int> row;
      for(size_t wi = 0; wi < bp::len(ex); wi++) {
        size_t token = (size_t)bp::extract<int>(ex[wi]);
        if(token < T) {
          row.push_back(token);
        }
      }
      ret.push_back(row);
      new_label.push_back(y);
    }
    return make_pair(ret, new_label);
  }
};


#endif
