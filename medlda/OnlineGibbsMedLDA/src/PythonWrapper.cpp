#include "OnlineGibbsMedLDAWrapper.h"
#include "OnlineGibbsMedLDAmtWrapper.h"


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
    .def("topWords", &paMedLDAgibbsWrapper::topWords)
    .def("numWord", &paMedLDAgibbsWrapper::numWord)
    .def("numLabel", &paMedLDAgibbsWrapper::numLabel)
    ;

  class_<paMedLDAgibbsMTWrapper>("paMedLDAgibbsMT",init<boost::python::dict>())
    .def("train", &paMedLDAgibbsMTWrapper::train)
    .def("infer", &paMedLDAgibbsMTWrapper::infer)
    .def("timeElapsed", &paMedLDAgibbsMTWrapper::timeElapsed)
    .def("testF1", &paMedLDAgibbsMTWrapper::testF1)
    .def("testRecall", &paMedLDAgibbsMTWrapper::testRecall)
    .def("testPrecision", &paMedLDAgibbsMTWrapper::testPrecision)
    .def("numWord", &paMedLDAgibbsMTWrapper::numWord)
    .def("numLabel", &paMedLDAgibbsMTWrapper::numLabel)
    ;
};



