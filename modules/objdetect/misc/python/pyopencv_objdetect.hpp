#ifdef HAVE_OPENCV_OBJDETECT

#include "opencv2/objdetect.hpp"

typedef QRCodeEncoder::Params QRCodeEncoder_Params;

typedef HOGDescriptor::HistogramNormType HOGDescriptor_HistogramNormType;
typedef HOGDescriptor::DescriptorStorageFormat HOGDescriptor_DescriptorStorageFormat;

static PyObject* pyopencv_cv_GraphicalCodeDetector_detectAndDecode(PyObject* self, PyObject* py_args, PyObject* kw);

static PyObject* pyopencv_cv_GraphicalCodeDetector_detectAndDecodeECI(PyObject* self, PyObject* py_args, PyObject* kw) {
    // Run original method
    PyObject* retval = pyopencv_cv_GraphicalCodeDetector_detectAndDecode(self, py_args, kw);

    if (PyErr_Occurred()) {
        PyObject *type = nullptr, *value = nullptr, *traceback = nullptr;
        PyErr_Fetch(&type, &value, &traceback);

        PyObject* object = PyUnicodeDecodeError_GetObject(value);
        if (object && PyBytes_Check(object)) {
            // TODO: use getEncoding. For now return just bytes array.
            PyTuple_SetItem(retval, 0, object);
            // PyObject* decoded = PyUnicode_FromEncodedObject(object, "ISO-8859-1", NULL);
            // PyTuple_SetItem(retval, 0, decoded);
        } else {
            PyErr_Restore(type, value, traceback);
        }
    }
    return retval;
}

// TODO: copy docstring somehow
#define PYOPENCV_EXTRA_METHODS_GraphicalCodeDetector \
    {"detectAndDecode", CV_PY_FN_WITH_KW_(pyopencv_cv_GraphicalCodeDetector_detectAndDecodeECI, 0), "detectAndDecode(img[, points[, straight_code]]) -> retval, points, straight_code\n.   @brief Both detects and decodes graphical code\n.   \n.        @param img grayscale or color (BGR) image containing graphical code.\n.        @param points optional output array of vertices of the found graphical code quadrangle, will be empty if not found.\n.        @param straight_code The optional output image containing binarized code"},

#endif
