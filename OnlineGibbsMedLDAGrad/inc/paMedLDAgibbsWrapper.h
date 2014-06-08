#ifndef PY_PAMEDLDA_AVE_WRAPPER_H
#define PY_PAMEDLDA_AVE_WRAPPER_H

#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/object.hpp>
#include <boost/python/str.hpp>
#include <boost/python/list.hpp>
#include <boost/python/dict.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <cmath>
#include <algorithm>

#include "debug.h" 
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

struct paMedLDAgibbsWrapper {
	paMedLDAgibbsWrapper(boost::python::dict config);
	~paMedLDAgibbsWrapper();

	void train(bp::object num_iter);
	void infer(bp::object num_test_sample);
	bp::object timeElapsed() const;
	bp::object testAcc() const {return bp::object(m_test_acc); }
	bp::list topicMatrix(bp::object category_no) const;
	bp::list topWords(bp::object category_no, int topk) const;
	bp::list topicDistOfInference(bp::object category_no) const;
	bp::list labelOfInference() const;

	// boost::python::array getTopWords();
	vector<shared_ptr<HybridMedLDA> > pamedlda;
	shared_ptr<Corpus> corpus;

	double m_test_acc;
};


#endif
