#include "OnlineGibbsMedLDAmtWrapper.h"
#include "stdio.h"

using namespace pyutils;
using namespace std;

paMedLDAgibbsMTWrapper::paMedLDAgibbsMTWrapper(boost::python::dict config) 
{
  this->_numLabel = bp::extract<int>(config["#label"]);
  this->_numWord = bp::extract<int>(config["#word"]);

  vector<int> category;
  for(size_t ci = 0; ci < this->_numLabel; ci++) {
    category.push_back(ci);
  }
  pamedlda = shared_ptr<OnlineGibbsMedLDA>(new OnlineGibbsMedLDA(category));

  /* hyper-parameters */
  pamedlda->K = bp::extract<int>(config["#topic"]);
  pamedlda->T = _numWord;
  pamedlda->num_category = _numLabel;
  pamedlda->init();

  /* optional parameters */
  if(config.has_key("alpha"))
    pamedlda->alpha0 = bp::extract<float>(config["alpha"]);
  if(config.has_key("beta"))
    pamedlda->beta0 = bp::extract<float>(config["beta"]);
  if(config.has_key("v"))
    pamedlda->v = bp::extract<float>(config["v"]);
  if(config.has_key("c"))
    pamedlda->c = bp::extract<float>(config["c"]);
  if(config.has_key("l"))
    pamedlda->l = bp::extract<float>(config["l"]);
  if(config.has_key("I"))
    pamedlda->I = bp::extract<int>(config["I"]);
  if(config.has_key("J"))
    pamedlda->J = bp::extract<int>(config["J"]);
  if(config.has_key("stepsize"))
    pamedlda->stepsize = bp::extract<double>(config["stepsize"]);
}

paMedLDAgibbsMTWrapper::~paMedLDAgibbsMTWrapper() 
{
}


void paMedLDAgibbsMTWrapper::train(bp::list batch, bp::list label) {
  auto filtered = filterWordAndLabel(batch, label, _numWord, _numLabel);
  pamedlda->train(filtered.first, filtered.second);
}

bp::list paMedLDAgibbsMTWrapper::infer(bp::list batch, bp::list _labels, boost::python::object num_test_sample, bool point_estimate) {
  auto filtered = filterWordAndLabel(batch, _labels, _numWord, _numLabel);
  auto docs = filtered.first;
  auto labels = filtered.second;
  vector<thread> threads(_numLabel);
  vector<vector<double> > my(_numLabel);
  pamedlda->point_estimate_for_test = point_estimate;
  vec2D<double> my_mt = pamedlda->inference(docs, bp::extract<int>(num_test_sample));

  int overlap = 0, prec_all = 0, recall_all = 0;
  bp::list ret;
  for(int i = 0; i < labels.size(); i++) {
    bp::list ret_labels;
    for(int ci = 0; ci < _numLabel; ci++) {  
      if(my_mt[i][ci] >= 0) {
        ret_labels.append(ci);
        prec_all++;
      }
      if(labels[i][ci] == 1) recall_all++;
      if(my_mt[i][ci] >= 0 and labels[i][ci] == 1) overlap++;
    }
    ret.append(ret_labels);
  }
  m_test_prec = overlap/(double)prec_all;
  m_test_recall = overlap/(double)recall_all;
  m_test_f1 = 2 * m_test_prec * m_test_recall / (m_test_prec + m_test_recall);
  return ret;
}

bp::object paMedLDAgibbsMTWrapper::timeElapsed() const {
  return bp::object(pamedlda->train_time);
}

bp::object paMedLDAgibbsMTWrapper::numWord() const {
  return bp::object(_numWord);
}

bp::object paMedLDAgibbsMTWrapper::numLabel() const {
  return bp::object(_numLabel);
}


