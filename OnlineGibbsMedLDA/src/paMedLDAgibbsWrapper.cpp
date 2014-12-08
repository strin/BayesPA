#include "paMedLDAgibbsWrapper.h"

using namespace pyutils;
using namespace std;

paMedLDAgibbsWrapper::paMedLDAgibbsWrapper(boost::python::dict config) 
{
  string train_file = bp::extract<string>(config["train_file"]);
  string test_file = bp::extract<string>(config["test_file"]);
  corpus = shared_ptr<Corpus>(new Corpus());
  corpus->loadDataGML(train_file, test_file);
  pamedlda.resize(corpus->newsgroupN);
  for(int ci = 0; ci < corpus->newsgroupN; ci++) {
    pamedlda[ci] = shared_ptr<HybridMedLDA>(new HybridMedLDA(&*corpus, ci));
    pamedlda[ci]->K = bp::extract<int>(config["num_topic"]);
    pamedlda[ci]->batchSize = bp::extract<int>(config["batchsize"]);
    pamedlda[ci]->lets_multic = true;
    pamedlda[ci]->alpha0 = bp::extract<float>(config["alpha"]);
    pamedlda[ci]->beta0 = bp::extract<float>(config["beta"]);
    pamedlda[ci]->c = bp::extract<float>(config["c"]);
    pamedlda[ci]->l = bp::extract<float>(config["l"]);
    pamedlda[ci]->I = bp::extract<int>(config["I"]);
    pamedlda[ci]->J = bp::extract<int>(config["J"]);
    pamedlda[ci]->init();
  }
}

paMedLDAgibbsWrapper::~paMedLDAgibbsWrapper() 
{

}


void paMedLDAgibbsWrapper::train(bp::list batch, bp::list label) {
  auto filtered = filterWordAndLabel(batch, label, _numWord(), _numLabel());
  int num_category = pamedlda[0]->num_category;
  vector<thread> threads(num_category);
  for(int ci = 0; ci < num_category; ci++) {
    threads[ci] = std::thread([&](int id)  {
      pamedlda[id]->train(filtered.first, filtered.second);
    }, ci);
  }
  for(int ci = 0; ci < num_category; ci++) threads[ci].join();
}

bp::list paMedLDAgibbsWrapper::infer(bp::list batch, bp::list label, boost::python::object num_test_sample) {
  int num_category = pamedlda[0]->num_category;
  auto filtered = filterWordAndLabel(batch, label, _numWord(), _numLabel());
  auto docs = filtered.first;
  auto labels = filtered.second;
  vector<thread> threads(num_category);
  vector<vector<double> > my(num_category);
  for(int ci = 0; ci < num_category; ci++) {
    threads[ci] = std::thread([&](int id)  {
      my[id] = pamedlda[id]->inference(docs, bp::extract<int>(num_test_sample));    
    }, ci);
  }
  for(int ci = 0; ci < num_category; ci++) threads[ci].join();  
  double acc = 0;
  bp::list ret;
  for(size_t d = 0; d < labels->size(); d++) {
    int output = 0;
    double confidence = 0-INFINITY;
    for( int ci = 0; ci < num_category; ci++) {
      if(my[ci][d] > confidence) {
        output = ci;
        confidence = pamedlda[ci]->testData->my[d];
      }
    }
    ret.append(output);
    if(output == (*labels)[d]) {
      acc++;
    }
  }
  m_test_acc = (double)acc/(double)labels->size();
  return ret;
}

bp::object paMedLDAgibbsWrapper::timeElapsed() const {
  double train_time = 0;
  for(auto t : pamedlda) {
    train_time += t->train_time;
  }
  return bp::object(train_time/(double)corpus->newsgroupN);
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

bp::list paMedLDAgibbsWrapper::labelOfInference() const {
  bp::list list;
  for(int d = 0; d < this->corpus->testDataSize; d++) {
    list.append(corpus->testData[d]->label);
  }
  return list;
}

inline bp::object paMedLDAgibbsWrapper::numWord() const {
  return bp::object(_numWord());
}

inline bp::object paMedLDAgibbsWrapper::numLabel() const {
  return bp::object(_numLabel());
}

inline size_t paMedLDAgibbsWrapper::_numWord() const {
  return pamedlda[0]->T;
}

inline size_t paMedLDAgibbsWrapper::_numLabel() const {
  return pamedlda[0]->num_category;
}

////////////////////////////////////////////////////////
//////////// private methods /////////////////////////
pair<vec2D<int>, vec<int> > 
paMedLDAgibbsWrapper::filterWordAndLabel(bp::list batch, bp::list label, size_t T, size_t C) {
  auto ret = makeVector2D<int>();
  auto new_label = makeVector<int>();
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
    ret->push_back(row);
    new_label->push_back(y);
  }
  return make_pair(ret, new_label);
}

BOOST_PYTHON_MODULE(libbayespagibbs)
{
  using namespace boost::python;
  
  class_<paMedLDAgibbsWrapper>("paMedLDAgibbs",init<boost::python::dict>())
    .def("train", &paMedLDAgibbsWrapper::train)
    .def("infer", &paMedLDAgibbsWrapper::infer)
    .def("timeElapsed", &paMedLDAgibbsWrapper::timeElapsed)
    .def("testAcc", &paMedLDAgibbsWrapper::testAcc)
    .def("topicMatrix", &paMedLDAgibbsWrapper::topicMatrix)
    .def("topicDistOfInference", &paMedLDAgibbsWrapper::topicDistOfInference)
    .def("labelOfInference", &paMedLDAgibbsWrapper::labelOfInference)
    .def("topWords", &paMedLDAgibbsWrapper::topWords)
    .def("numWord", &paMedLDAgibbsWrapper::numWord)
    .def("numLabel", &paMedLDAgibbsWrapper::numLabel)
    ;
};


