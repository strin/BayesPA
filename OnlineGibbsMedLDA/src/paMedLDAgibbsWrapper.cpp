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


void paMedLDAgibbsWrapper::train(bp::list batch, bp::list labels) {
	auto batch_vec = filterWord(batch, _numWord());
	vector<thread> threads(corpus->newsgroupN);
	for(int ci = 0; ci < corpus->newsgroupN; ci++) {
		threads[ci] = std::thread([&](int id)  {
			pamedlda[id]->train(batch_vec);
		}, ci);
	}
	for(int ci = 0; ci < corpus->newsgroupN; ci++) threads[ci].join();
}

void paMedLDAgibbsWrapper::infer(boost::python::object num_test_sample) {
	vector<thread> threads(corpus->newsgroupN);
	for(int ci = 0; ci < corpus->newsgroupN; ci++) {
		threads[ci] = std::thread([&](int id)  {
			pamedlda[id]->inference(pamedlda[id]->testData, 
							bp::extract<int>(num_test_sample));		
		}, ci);
	}
	for(int ci = 0; ci < corpus->newsgroupN; ci++) threads[ci].join();	
	double acc = 0;
	for( int d = 0; d < pamedlda[0]->testData->D; d++) {
		int label;
		double confidence = 0-INFINITY;
		for( int ci = 0; ci < corpus->newsgroupN; ci++) {
			if(pamedlda[ci]->testData->my[d] > confidence) {
				label = ci;
				confidence = pamedlda[ci]->testData->my[d];
			}
		}
		if(pamedlda[label]->testData->y[d] == 1) {
			acc++;
		}
	}
	m_test_acc = (double)acc/(double)pamedlda[0]->testData->D;	
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
	if(pamedlda[ci]->Zbar_test == NULL) return mat;
	for(int d = 0; d < this->pamedlda[ci]->testData->D; d++) {
		bp::list row;
		for(int k = 0; k < this->pamedlda[ci]->K; k++) {
			row.append(pamedlda[ci]->Zbar_test[d][k]);
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
	return pamedlda[0]->category;
}

////////////////////////////////////////////////////////
//////////// private methods /////////////////////////
vec2D<int> paMedLDAgibbsWrapper::filterWord(bp::list batch, size_t T) {
	auto ret = makeVector2D<int>();
	for(size_t ni = 0; ni < bp::len(batch); ni++) {
		bp::list ex = bp::extract<bp::list>(batch[ni]);
		std::vector<int> row;
		for(size_t wi = 0; wi < bp::len(ex); wi++) {
			size_t token = (size_t)bp::extract<int>(ex[wi]);
			if(token < T) {
				row.push_back(token);
			}
		}
		ret->push_back(row);
	}
	return ret;
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


