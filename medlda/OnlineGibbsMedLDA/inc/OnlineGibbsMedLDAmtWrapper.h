#ifndef PY_PAMEDLDA_GIBBS_MT_WRAPPER_H
#define PY_PAMEDLDA_GIBBS_MT_WRAPPER_H

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

class paMedLDAgibbsMTWrapper {
public:
	paMedLDAgibbsMTWrapper(boost::python::dict config);
	~paMedLDAgibbsMTWrapper();

	void train(bp::list batch, bp::list label);
	bp::list infer(bp::list batch, bp::list labels, bp::object num_test_sample, bool point_estimate);

	bp::object timeElapsed() const;

        bp::object testF1() const {
          return bp::object(m_test_f1);
        }

        bp::object testPrecision() const {
          return bp::object(m_test_prec);
        }

        bp::object testRecall() const {
          return bp::object(m_test_recall);
        }

	shared_ptr<OnlineGibbsMedLDA> pamedlda; // shared model.

	bp::object numWord() const;
	bp::object numLabel() const;

	double m_test_f1, m_test_prec, m_test_recall;

private:
  /* return number of words in the vocabulary */
  size_t _numWord;
  size_t _numLabel;

  static pair<vec2D<int>, vec2D<int> > filterWordAndLabel(bp::list batch, bp::list label, size_t T, size_t C) {
    auto ret = vec2D<int>();
    vec2D<int> new_labels;
    for(size_t ni = 0; ni < bp::len(batch); ni++) {
      bp::list ex = bp::extract<bp::list>(batch[ni]);
      vector<int> labels;
      bp::list label_list = bp::extract<bp::list>(label[ni]);
      for(size_t ci = 0; ci < bp::len(label_list); ci++) {
        int label = (int)bp::extract<int>(label_list[ci]);
        if(label >= C) continue;
        labels.push_back(label);
      }

      std::vector<int> row;
      for(size_t wi = 0; wi < bp::len(ex); wi++) {
        size_t token = (size_t)bp::extract<int>(ex[wi]);
        if(token < T) {
          row.push_back(token);
        }
      }

      ret.push_back(row);
      new_labels.push_back(labels);
    }
    return make_pair(ret, new_labels);
  }
};


#endif
