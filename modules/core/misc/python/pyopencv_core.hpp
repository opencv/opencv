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

#define PYOPENCV_EXTRA_METHODS_CV \
  {"CV_MAKETYPE", CV_PY_FN_WITH_KW(pycvMakeType), "CV_MAKETYPE(depth, channels) -> retval"},

#endif  // HAVE_OPENCV_CORE
#endif  // OPENCV_CORE_PYOPENCV_CORE_HPP
