#ifndef OPENCV_CORE_PYOPENCV_CORE_HPP
#define OPENCV_CORE_PYOPENCV_CORE_HPP

#ifdef HAVE_OPENCV_CORE

static PyObject* pycvMakeType(PyObject* , PyObject* args, PyObject* kw) {
    const char *keywords[] = { "depth", "channels", NULL };

    int depth, channels;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "ii", (char**)keywords, &depth, &channels))
        return NULL;

    int type = CV_MAKETYPE(depth, channels);
    return PyInt_FromLong(type);
}

template <int depth>
static PyObject* pycvMakeTypeCh(PyObject*, PyObject *value) {
    int channels = (int)PyLong_AsLong(value);
    return PyInt_FromLong(CV_MAKETYPE(depth, channels));
}

#define PYOPENCV_EXTRA_METHODS_CV \
  {"CV_MAKETYPE", CV_PY_FN_WITH_KW(pycvMakeType), "CV_MAKETYPE(depth, channels) -> retval"}, \
  {"CV_8UC", (PyCFunction)(pycvMakeTypeCh<CV_8U>), METH_O, "CV_8UC(channels) -> retval"}, \
  {"CV_8SC", (PyCFunction)(pycvMakeTypeCh<CV_8S>), METH_O, "CV_8SC(channels) -> retval"}, \
  {"CV_16UC", (PyCFunction)(pycvMakeTypeCh<CV_16U>), METH_O, "CV_16UC(channels) -> retval"}, \
  {"CV_16SC", (PyCFunction)(pycvMakeTypeCh<CV_16S>), METH_O, "CV_16SC(channels) -> retval"}, \
  {"CV_32SC", (PyCFunction)(pycvMakeTypeCh<CV_32S>), METH_O, "CV_32SC(channels) -> retval"}, \
  {"CV_32FC", (PyCFunction)(pycvMakeTypeCh<CV_32F>), METH_O, "CV_32FC(channels) -> retval"}, \
  {"CV_64FC", (PyCFunction)(pycvMakeTypeCh<CV_64F>), METH_O, "CV_64FC(channels) -> retval"}, \
  {"CV_16FC", (PyCFunction)(pycvMakeTypeCh<CV_16F>), METH_O, "CV_16FC(channels) -> retval"},

#endif  // HAVE_OPENCV_CORE
#endif  // OPENCV_CORE_PYOPENCV_CORE_HPP
