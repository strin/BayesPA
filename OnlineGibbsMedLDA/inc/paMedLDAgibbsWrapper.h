#ifndef PY_PAMEDLDA_AVE_WRAPPER_H
#define PY_PAMEDLDA_AVE_WRAPPER_H

#include <iostream>
#include <vector>
#include <thread>
#include <cmath>
#include <algorithm>

#include "debug.h" 
#include "pyutils.h"
#include "HybridMedLDA.h"

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
	bp::list infer(bp::list batch, bp::list label, bp::object num_test_sample);
	bp::object timeElapsed() const;
	bp::object testAcc() const {return bp::object(m_test_acc); }
	bp::list topicMatrix(bp::object category_no) const;
	bp::list topWords(bp::object category_no, int topk) const;
	bp::list topicDistOfInference(bp::object category_no) const;

	vector<shared_ptr<HybridMedLDA> > pamedlda;

	bp::object numWord() const;
	bp::object numLabel() const;

	double m_test_acc;

private:
  /* filter out tokens that are not in range [0, T-1], and documents whose label not in [0, C-1]. */
  std::pair<pyutils::vec2D<int>, pyutils::vec<int> >
  filterWordAndLabel(bp::list batch, bp::list label, size_t T, size_t C);

  /* return number of words in the vocabulary */
  size_t _numWord;
  size_t _numLabel;
};


#endif
