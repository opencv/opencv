#ifdef HAVE_OPENCV_OBJDETECT

#include "opencv2/objdetect.hpp"

typedef QRCodeEncoder::Params QRCodeEncoder_Params;

typedef HOGDescriptor::HistogramNormType HOGDescriptor_HistogramNormType;
typedef HOGDescriptor::DescriptorStorageFormat HOGDescriptor_DescriptorStorageFormat;

static const char* eciString(QRCodeEncoder::ECIEncodings eci) {
    switch (eci) {
    case QRCodeEncoder::ECIEncodings::ECI_UTF8:
        return "utf-8";
    case QRCodeEncoder::ECIEncodings::ECI_SHIFT_JIS:
        return "shift-jis";
    }
}

static PyObject* pyopencv_cv_GraphicalCodeDetector_detectAndDecode(PyObject* self, PyObject* py_args, PyObject* kw);

static PyObject* pyopencv_cv_GraphicalCodeDetector_detectAndDecodeECI(PyObject* self, PyObject* py_args, PyObject* kw) {
    // Run original method
    PyObject* retval = pyopencv_cv_GraphicalCodeDetector_detectAndDecode(self, py_args, kw);

    cv::GraphicalCodeDetector* obj = 0;
    pyopencv_GraphicalCodeDetector_getp(self, obj);
    // if (obj->getEncoding() != ECI_UTF8) {

    // }

    if (PyErr_Occurred()) {
        PyObject *type = nullptr, *value = nullptr, *traceback = nullptr;
        PyErr_Fetch(&type, &value, &traceback);

        PyObject* object = PyUnicodeDecodeError_GetObject(value);
        if (object && PyBytes_Check(object)) {
            // PyTuple_SetItem(retval, 0, object);
            const char* encoding = eciString(QRCodeEncoder::ECIEncodings::ECI_SHIFT_JIS);
            PyObject* decoded = PyUnicode_FromEncodedObject(object, encoding, NULL);
            PyTuple_SetItem(retval, 0, decoded);
        } else {
            PyErr_Restore(type, value, traceback);
        }
    }
    return retval;
}

// TODO: copy docstring somehow
#define PYOPENCV_EXTRA_METHODS_GraphicalCodeDetector \
    {"detectAndDecode", CV_PY_FN_WITH_KW_(pyopencv_cv_GraphicalCodeDetector_detectAndDecodeECI, 0), ""},

#endif
