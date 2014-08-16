#if defined(_MSC_VER) && (_MSC_VER >= 1800)
// eliminating duplicated round() declaration
#define HAVE_ROUND
#endif

#include <Python.h>

#define MODULESTR "cv2"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include "pyopencv_generated_include.h"
#include "opencv2/core/types_c.h"
#include "pyopencv_generated_typedefs.h"

#include "pycv2.hpp"
////////////////////////////////////////////////////////////////////////////////////////////////////

static void OnMouse(int event, int x, int y, int flags, void* param)
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    PyObject *o = (PyObject*)param;
    PyObject *args = Py_BuildValue("iiiiO", event, x, y, flags, PyTuple_GetItem(o, 1));

    PyObject *r = PyObject_Call(PyTuple_GetItem(o, 0), args, NULL);
    if (r == NULL)
        PyErr_Print();
    else
        Py_DECREF(r);
    Py_DECREF(args);
    PyGILState_Release(gstate);
}

static PyObject *pycvSetMouseCallback(PyObject*, PyObject *args, PyObject *kw)
{
    const char *keywords[] = { "window_name", "on_mouse", "param", NULL };
    char* name;
    PyObject *on_mouse;
    PyObject *param = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "sO|O", (char**)keywords, &name, &on_mouse, &param))
        return NULL;
    if (!PyCallable_Check(on_mouse)) {
        PyErr_SetString(PyExc_TypeError, "on_mouse must be callable");
        return NULL;
    }
    if (param == NULL) {
        param = Py_None;
    }
    ERRWRAP2(setMouseCallback(name, OnMouse, Py_BuildValue("OO", on_mouse, param)));
    Py_RETURN_NONE;
}

static void OnChange(int pos, void *param)
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    PyObject *o = (PyObject*)param;
    PyObject *args = Py_BuildValue("(i)", pos);
    PyObject *r = PyObject_Call(PyTuple_GetItem(o, 0), args, NULL);
    if (r == NULL)
        PyErr_Print();
    Py_DECREF(args);
    PyGILState_Release(gstate);
}

static PyObject *pycvCreateTrackbar(PyObject*, PyObject *args)
{
    PyObject *on_change;
    char* trackbar_name;
    char* window_name;
    int *value = new int;
    int count;

    if (!PyArg_ParseTuple(args, "ssiiO", &trackbar_name, &window_name, value, &count, &on_change))
        return NULL;
    if (!PyCallable_Check(on_change)) {
        PyErr_SetString(PyExc_TypeError, "on_change must be callable");
        return NULL;
    }
    ERRWRAP2(createTrackbar(trackbar_name, window_name, value, count, OnChange, Py_BuildValue("OO", on_change, Py_None)));
    Py_RETURN_NONE;
}

///////////////////////////////////////////////////////////////////////////////////////

static int convert_to_char(PyObject *o, char *dst, const char *name = "no_name")
{
  if (PyString_Check(o) && PyString_Size(o) == 1) {
    *dst = PyString_AsString(o)[0];
    return 1;
  } else {
    (*dst) = 0;
    return failmsg("Expected single character string for argument '%s'", name);
  }
}

#if PY_MAJOR_VERSION >= 3
#define MKTYPE2(NAME) pyopencv_##NAME##_specials(); if (!to_ok(&pyopencv_##NAME##_Type)) return NULL;
#else
#define MKTYPE2(NAME) pyopencv_##NAME##_specials(); if (!to_ok(&pyopencv_##NAME##_Type)) return
#endif

#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wunused-parameter"
#  pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif

#include "pyopencv_generated_types.h"
#include "pyopencv_generated_funcs.h"

static PyMethodDef methods[] = {

#include "pyopencv_generated_func_tab.h"
  {"createTrackbar", pycvCreateTrackbar, METH_VARARGS, "createTrackbar(trackbarName, windowName, value, count, onChange) -> None"},
  {"setMouseCallback", (PyCFunction)pycvSetMouseCallback, METH_VARARGS | METH_KEYWORDS, "setMouseCallback(windowName, onMouse [, param]) -> None"},
  {NULL, NULL},
};

/************************************************************************/
/* Module init */

static int to_ok(PyTypeObject *to)
{
  to->tp_alloc = PyType_GenericAlloc;
  to->tp_new = PyType_GenericNew;
  to->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  return (PyType_Ready(to) == 0);
}


#if PY_MAJOR_VERSION >= 3
extern "C" CV_EXPORTS PyObject* PyInit_cv2();
static struct PyModuleDef cv2_moduledef =
{
    PyModuleDef_HEAD_INIT,
    MODULESTR,
    "Python wrapper for OpenCV.",
    -1,     /* size of per-interpreter state of the module,
               or -1 if the module keeps state in global variables. */
    methods
};

PyObject* PyInit_cv2()
#else
extern "C" CV_EXPORTS void initcv2();

void initcv2()
#endif
{
  import_array();

#include "pyopencv_generated_type_reg.h"

#if PY_MAJOR_VERSION >= 3
  PyObject* m = PyModule_Create(&cv2_moduledef);
#else
  PyObject* m = Py_InitModule(MODULESTR, methods);
#endif
  PyObject* d = PyModule_GetDict(m);

  PyDict_SetItemString(d, "__version__", PyString_FromString(CV_VERSION));

  opencv_error = PyErr_NewException((char*)MODULESTR".error", NULL, NULL);
  PyDict_SetItemString(d, "error", opencv_error);

#define PUBLISH(I) PyDict_SetItemString(d, #I, PyInt_FromLong(I))
//#define PUBLISHU(I) PyDict_SetItemString(d, #I, PyLong_FromUnsignedLong(I))
#define PUBLISH2(I, value) PyDict_SetItemString(d, #I, PyLong_FromLong(value))

  PUBLISH(CV_8U);
  PUBLISH(CV_8UC1);
  PUBLISH(CV_8UC2);
  PUBLISH(CV_8UC3);
  PUBLISH(CV_8UC4);
  PUBLISH(CV_8S);
  PUBLISH(CV_8SC1);
  PUBLISH(CV_8SC2);
  PUBLISH(CV_8SC3);
  PUBLISH(CV_8SC4);
  PUBLISH(CV_16U);
  PUBLISH(CV_16UC1);
  PUBLISH(CV_16UC2);
  PUBLISH(CV_16UC3);
  PUBLISH(CV_16UC4);
  PUBLISH(CV_16S);
  PUBLISH(CV_16SC1);
  PUBLISH(CV_16SC2);
  PUBLISH(CV_16SC3);
  PUBLISH(CV_16SC4);
  PUBLISH(CV_32S);
  PUBLISH(CV_32SC1);
  PUBLISH(CV_32SC2);
  PUBLISH(CV_32SC3);
  PUBLISH(CV_32SC4);
  PUBLISH(CV_32F);
  PUBLISH(CV_32FC1);
  PUBLISH(CV_32FC2);
  PUBLISH(CV_32FC3);
  PUBLISH(CV_32FC4);
  PUBLISH(CV_64F);
  PUBLISH(CV_64FC1);
  PUBLISH(CV_64FC2);
  PUBLISH(CV_64FC3);
  PUBLISH(CV_64FC4);

#include "pyopencv_generated_const_reg.h"
#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}
