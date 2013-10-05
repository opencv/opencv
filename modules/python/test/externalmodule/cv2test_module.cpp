#include <Python.h>

#define MODULESTR "cv2test"

#include "cv2test.hpp"
#include "cv2.hpp"
#include "cv2support.cpp"

using namespace cv;
using namespace cv2test;

#include "cv2test_generated_types.h"
#include "cv2test_generated_funcs.h"

static PyMethodDef methods[] = {
#include "cv2test_generated_func_tab.h"
  {NULL, NULL, 0, NULL},
};


extern "C" CV_EXPORTS void initcv2test();

void initcv2test() {
  import_array();

  #include "cv2test_generated_type_reg.h"

  PyObject* m = Py_InitModule(MODULESTR, methods);
  PyObject* d = PyModule_GetDict(m);

  opencv_error = PyErr_NewException((char*)MODULESTR".error", NULL, NULL);
  PyDict_SetItemString(d, "error", opencv_error);

  #include "cv2test_generated_const_reg.h"

}
