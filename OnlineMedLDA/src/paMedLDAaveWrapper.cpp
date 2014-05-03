#include "paMedLDAaveWrapper.h"

paMedLDAaveWrapper::paMedLDAaveWrapper(boost::python::dict config) 
{
	string train_file = bp::extract<string>(config["train_file"]);
	string test_file = bp::extract<string>(config["test_file"]);
	corpus = shared_ptr<Corpus>(new Corpus());
	corpus->loadDataGML(train_file, test_file);
	pamedlda.resize(corpus->newsgroup_n);
	for(int ci = 0; ci < corpus->newsgroup_n; ci++) {
		pamedlda[ci] = shared_ptr<paMedLDAave>(new paMedLDAave(&*corpus, ci));
		pamedlda[ci]->m_K = bp::extract<int>(config["num_topic"]);
		pamedlda[ci]->m_batchsize = bp::extract<int>(config["batchsize"]);
		pamedlda[ci]->lets_multic = true;
		pamedlda[ci]->alpha = bp::extract<float>(config["alpha"]);
		pamedlda[ci]->beta = bp::extract<float>(config["beta"]);
		pamedlda[ci]->m_c = bp::extract<float>(config["c"]);
		pamedlda[ci]->m_l = bp::extract<float>(config["l"]);
		pamedlda[ci]->m_I = bp::extract<int>(config["I"]);
		pamedlda[ci]->m_J = bp::extract<int>(config["J"]);
		pamedlda[ci]->m_v = sqrt(bp::extract<float>(config["sigma2"]));
		pamedlda[ci]->m_epoch = bp::extract<int>(config["epoch"]);
		pamedlda[ci]->init();
	}
}

paMedLDAaveWrapper::~paMedLDAaveWrapper() 
{

}


void paMedLDAaveWrapper::train(boost::python::object num_iter) {
	vector<thread> threads(corpus->newsgroup_n);
	for(int ci = 0; ci < corpus->newsgroup_n; ci++) {
		threads[ci] = std::thread([&](int id)  {
			pamedlda[id]->train(bp::extract<int>(num_iter));		
		}, ci);
	}
	for(int ci = 0; ci < corpus->newsgroup_n; ci++) threads[ci].join();
}

void paMedLDAaveWrapper::infer(boost::python::object num_test_sample) {
	vector<thread> threads(corpus->newsgroup_n);
	for(int ci = 0; ci < corpus->newsgroup_n; ci++) {
		threads[ci] = std::thread([&](int id)  {
			pamedlda[id]->inference(pamedlda[id]->test_data, 
							bp::extract<int>(num_test_sample));		
		}, ci);
	}
	for(int ci = 0; ci < corpus->newsgroup_n; ci++) threads[ci].join();	
	double acc = 0;
	for( int d = 0; d < pamedlda[0]->test_data->D; d++) {
		int label;
		double confidence = 0-INFINITY;
		for( int ci = 0; ci < corpus->newsgroup_n; ci++) {
			if(pamedlda[ci]->local_test->my[d][0] > confidence) {
				label = ci;
				confidence = pamedlda[ci]->local_test->my[d][0];
			}
		}
		if(corpus->test_data.doc[d].y[0] == label) {
			acc++;
		}
	}
	m_test_acc = (double)acc/(double)pamedlda[0]->test_data->D;	
}

bp::object paMedLDAaveWrapper::timeElapsed() const {
	double train_time = 0;
	for(auto t : pamedlda) {
		train_time += t->train_time;
	}
	return bp::object(train_time/(double)corpus->newsgroup_n);
}


BOOST_PYTHON_MODULE(libbayespa)
{
  using namespace boost::python;
  
  class_<paMedLDAaveWrapper>("paMedLDAave",init<boost::python::dict>())
    .def("train", &paMedLDAaveWrapper::train)
    .def("infer", &paMedLDAaveWrapper::infer)
    .def("timeElapsed", &paMedLDAaveWrapper::timeElapsed)
    .def("testAcc", &paMedLDAaveWrapper::testAcc)
    ;
};


