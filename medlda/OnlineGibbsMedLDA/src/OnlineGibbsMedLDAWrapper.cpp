#include "OnlineGibbsMedLDAWrapper.h"
#include "stdio.h"

using namespace pyutils;
using namespace std;

paMedLDAgibbsWrapper::paMedLDAgibbsWrapper(boost::python::dict config) 
{
  this->_numLabel = bp::extract<int>(config["#label"]);
  this->_numWord = bp::extract<int>(config["#word"]);

  pamedlda.resize(_numLabel);
  
  for(int ci = 0; ci < _numLabel; ci++) {
    pamedlda[ci] = shared_ptr<OnlineGibbsMedLDA>(new OnlineGibbsMedLDA({ci}));

    /* hyper-parameters */
    pamedlda[ci]->K = bp::extract<int>(config["#topic"]);
    pamedlda[ci]->T = _numWord;
    pamedlda[ci]->num_category = _numLabel;
    pamedlda[ci]->init();

    /* optional parameters */
    if(config.has_key("alpha"))
      pamedlda[ci]->alpha0 = bp::extract<float>(config["alpha"]);
    if(config.has_key("beta"))
      pamedlda[ci]->beta0 = bp::extract<float>(config["beta"]);
    if(config.has_key("v"))
      pamedlda[ci]->v = bp::extract<float>(config["v"]);
    if(config.has_key("c"))
      pamedlda[ci]->c = bp::extract<float>(config["c"]);
    if(config.has_key("l"))
      pamedlda[ci]->l = bp::extract<float>(config["l"]);
    if(config.has_key("I"))
      pamedlda[ci]->I = bp::extract<int>(config["I"]);
    if(config.has_key("J"))
      pamedlda[ci]->J = bp::extract<int>(config["J"]);
    if(config.has_key("stepsize"))
      pamedlda[ci]->stepsize = bp::extract<double>(config["stepsize"]);
  }
}

paMedLDAgibbsWrapper::~paMedLDAgibbsWrapper() 
{
}


void paMedLDAgibbsWrapper::train(bp::list batch, bp::list label) {
  auto filtered = filterWordAndLabel(batch, label, _numWord, _numLabel);
  vector<thread> threads(_numLabel);
  for(int ci = 0; ci < _numLabel; ci++) {
    threads[ci] = std::thread([&](int id)  {
      pamedlda[id]->train(filtered.first, filtered.second);
    }, ci);
  }
  for(int ci = 0; ci < _numLabel; ci++) threads[ci].join();
}

bp::list paMedLDAgibbsWrapper::infer(bp::list batch, bp::list label, boost::python::object num_test_sample, bool point_estimate) {
  auto filtered = filterWordAndLabel(batch, label, _numWord, _numLabel);
  auto docs = filtered.first;
  auto labels = filtered.second;
  vector<thread> threads(_numLabel);
  vector<vector<double> > my(_numLabel);
  for(int ci = 0; ci < _numLabel; ci++) {
    threads[ci] = std::thread([&](int id)  {
      pamedlda[id]->point_estimate_for_test = point_estimate;
      vec2D<double> my_mt = pamedlda[id]->inference(docs, bp::extract<int>(num_test_sample));
      for(auto resp : my_mt) {
        my[id].push_back(resp[0]);
      }
    }, ci);
  }

  for(int ci = 0; ci < _numLabel; ci++) threads[ci].join();  
  double acc = 0;
  bp::list ret;
  for(size_t d = 0; d < labels.size(); d++) {
    int output = 0;
    double confidence = 0-INFINITY;
    for( int ci = 0; ci < _numLabel; ci++) {
      if(my[ci][d] > confidence) {
        output = ci;
        confidence = my[ci][d];
      }
    }
    ret.append(output);
    if(output == labels[d]) {
      acc++;
    }
  }
  m_test_acc = (double)acc/(double)labels.size();
  return ret;
}

bp::object paMedLDAgibbsWrapper::timeElapsed() const {
  double train_time = 0;
  for(auto t : pamedlda) {
    train_time += t->train_time;
  }
  return bp::object(train_time/(double)_numLabel);
}

bp::list paMedLDAgibbsWrapper::topicMatrix(bp::object category_no) const {
  int ci = bp::extract<int>(category_no);
  bp::list mat;
  for(int k = 0; k < this->pamedlda[ci]->K; k++) {
    bp::list row;
    for(int t = 0; t < this->pamedlda[ci]->T; t++) {
      row.append(pamedlda[ci]->gamma[k][t]/(double)pamedlda[ci]->gammasum[k]);
    }
    mat.append(row);
  }
  return mat;
}

bp::list paMedLDAgibbsWrapper::topWords(bp::object category_no, int topk) const {
  int ci = bp::extract<int>(category_no);
  bp::list mat;
  for(int k = 0; k < this->pamedlda[ci]->K; k++) {
    vector<sortable> v;
    for(int t = 0; t < this->pamedlda[ci]->T; t++) {
      sortable x;
      x.value = pamedlda[ci]->gamma[k][t]/(double)pamedlda[ci]->gammasum[k];
      x.index = t;
      v.push_back(x);
    }
    sort(v.begin(), v.end());
    bp::list row;
    for(int i = 0; i < topk; i++) {
      row.append(v[i].index);
    }
    mat.append(row);
  }
  return mat;  
}

bp::list paMedLDAgibbsWrapper::topicDistOfInference(bp::object category_no) const {
  int ci = bp::extract<int>(category_no);
  bp::list mat;
  for(const vector<double>& zbar_row : this->pamedlda[ci]->Zbar_test) {
    bp::list row;
    for(const double& zbar_entry : zbar_row) {
      row.append(zbar_entry);
    }
    mat.append(row);
  }
  return mat;
}

bp::object paMedLDAgibbsWrapper::numWord() const {
  return bp::object(_numWord);
}

bp::object paMedLDAgibbsWrapper::numLabel() const {
  return bp::object(_numLabel);
}

