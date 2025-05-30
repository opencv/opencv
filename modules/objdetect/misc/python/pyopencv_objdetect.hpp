#ifdef HAVE_OPENCV_OBJDETECT

#include "opencv2/objdetect.hpp"

typedef QRCodeEncoder::Params QRCodeEncoder_Params;

typedef HOGDescriptor::HistogramNormType HOGDescriptor_HistogramNormType;
typedef HOGDescriptor::DescriptorStorageFormat HOGDescriptor_DescriptorStorageFormat;

static PyObject* pyopencv_cv_GraphicalCodeDetector_detectAndDecode(PyObject* self, PyObject* py_args, PyObject* kw);

static PyObject* detectAndDecodeBytes(PyObject* self, PyObject* py_args, PyObject* kw) {
    // Run original method
    PyObject* retval = pyopencv_cv_GraphicalCodeDetector_detectAndDecode(self, py_args, kw);

    if (PyErr_Occurred()) {
        PyObject *type = nullptr, *value = nullptr, *traceback = nullptr;
        PyErr_Fetch(&type, &value, &traceback);

        PyObject* object = PyUnicodeDecodeError_GetObject(value);
        if (object && PyBytes_Check(object)) {
            PyTuple_SetItem(retval, 0, object);
        } else {
            PyErr_Restore(type, value, traceback);
        }
    } else {
        PyObject* str = PyTuple_GetItem(retval, 0);
        PyObject* bytes = PyUnicode_AsEncodedString(str, "utf-8", 0);
        PyTuple_SetItem(retval, 0, bytes);
    }
    return retval;
}

// TODO: copy docstring somehow
#define PYOPENCV_EXTRA_METHODS_GraphicalCodeDetector \
    {"detectAndDecodeBytes", CV_PY_FN_WITH_KW_(detectAndDecodeBytes, 0), ""},

#endif
