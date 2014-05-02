#ifndef PY_PAMEDLDA_AVE_WRAPPER_H
#define PY_PAMEDLDA_AVE_WRAPPER_H

#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/object.hpp>
#include <boost/python/str.hpp>
#include <boost/python/list.hpp>
#include <boost/python/dict.hpp>

#include "paMedLDAave.h"


using std::shared_ptr;

struct paMedLDAaveWrapper {
	paMedLDAaveWrapper(boost::python::dict config);
	~paMedLDAaveWrapper();

	void train(boost::python::object num_iter);
	void infer(boost::python::object data_path);

	boost::python::array getTopWords();

	shared_ptr<paMedLDAave> pamedlda;
};

#endif