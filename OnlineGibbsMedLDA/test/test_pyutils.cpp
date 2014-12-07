#include "pyutils.h"

using namespace pyutils;
using namespace std;

void test_makeVector() {
  bp::list list;
  list.append<int>(1.0);
  list.append(2.0);
  list.append(3.0);
  
  auto vector = makeVector<double>(list);
  for(auto x : *vector) {
    cout << x << " ";
  }
  cout << endl;
}

void test_makeVector2D() {
  bp::list list, list1, list2;
  list1.append(1.0);
  list1.append(2.0);
  list2.append(3.0);
  list2.append(4.0);
  list.append(list1);
  list.append(list2);
  
  auto vector2D = makeVector2D<double>(list);
  for(auto& vector : *vector2D) {
    for(auto& x : vector) {
      cout << x << " ";
    }
    cout << endl;
  }
  cout << endl;
}

int main() {
  Py_Initialize(); 

  // test_makeVector2D();

  Py_Finalize(); 
  return 0;
}
