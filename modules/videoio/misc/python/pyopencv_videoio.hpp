#ifdef HAVE_OPENCV_VIDEOIO
typedef std::vector<VideoCaptureAPIs> vector_VideoCaptureAPIs;
typedef std::vector<VideoCapture> vector_VideoCapture;

template<> struct pyopencvVecConverter<cv::VideoCaptureAPIs>
{
    static bool to(PyObject* obj, std::vector<cv::VideoCaptureAPIs>& value, const ArgInfo& info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const std::vector<cv::VideoCaptureAPIs>& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

template<>
bool pyopencv_to(PyObject *o, std::vector<cv::VideoCaptureAPIs>& apis, const ArgInfo& info)
{
  return pyopencvVecConverter<cv::VideoCaptureAPIs>::to(o, apis, info);
}

template<> bool pyopencv_to(PyObject* obj, cv::VideoCapture& stream, const ArgInfo& info)
{
    Ptr<VideoCapture> * obj_getp = nullptr;
    if (!pyopencv_VideoCapture_getp(obj, obj_getp))
        return (failmsgp("Incorrect type of self (must be 'VideoCapture' or its derivative)") != nullptr);

    stream = **obj_getp;
    return true;
}

class PythonStreamReader : public cv::IStreamReader
{
public:
    PythonStreamReader(PyObject* _obj = nullptr) : obj(_obj)
    {
        if (obj)
            Py_INCREF(obj);
    }

    ~PythonStreamReader()
    {
        PyEnsureGIL gil;
        if (obj)
            Py_DECREF(obj);
    }

    long long read(char* buffer, long long size) CV_OVERRIDE
    {
        if (!obj)
            return 0;

        PyEnsureGIL gil;
        PySafeObject res(PyObject_CallMethod(obj, "read", "L", size));
        if (!res)
        {
            PyErr_Clear();
            CV_Error(cv::Error::StsError, "Python .read() call error");
        }

        char* src = NULL;
        Py_ssize_t len = 0;
        if (PyBytes_AsStringAndSize(res, &src, &len) < 0)
        {
            PyErr_Clear();
            CV_Error(cv::Error::StsError, "Python buffer access error");
        }

        CV_CheckLE(static_cast<size_t>(len), static_cast<size_t>(size), "Stream chunk size should be less or equal than requested size");
        std::memcpy(buffer, src, static_cast<size_t>(len));
        return len;
    }

    long long seek(long long offset, int way) CV_OVERRIDE
    {
        if (!obj)
            return 0;

        PyEnsureGIL gil;
        PySafeObject res(PyObject_CallMethod(obj, "seek", "Li", offset, way));
        if (!res)
        {
            PyErr_Clear();
            CV_Error(cv::Error::StsError, "Python .seek() call error");
        }

        long long pos = PyLong_AsLongLong(res);
        if (PyErr_Occurred())
        {
            PyErr_Clear();
            CV_Error(cv::Error::StsError, "Python .seek() result => long long conversion error");
        }
        return pos;
    }

private:
    PyObject* obj;
};

template<>
bool pyopencv_to(PyObject* obj, Ptr<cv::IStreamReader>& p, const ArgInfo&)
{
    if (!obj)
        return false;

    PyObject* ioModule = PyImport_ImportModule("io");
    PyObject* type = PyObject_GetAttrString(ioModule, "BufferedIOBase");
    Py_DECREF(ioModule);
    bool isValidPyType = PyObject_IsInstance(obj, type) == 1;
    Py_DECREF(type);

    if (!isValidPyType)
    {
        PyErr_SetString(PyExc_TypeError, "Input stream should be derived from io.BufferedIOBase");
        return false;
    }

    if (!PyErr_Occurred()) {
        p = makePtr<PythonStreamReader>(obj);
        return true;
    }
    return false;
}

static PyObject* pycvVideoEnter(PyObject* self, PyObject* /*args*/, PyObject* /*kw*/) {
    Py_INCREF(self);
    return self;
}

static PyObject* pycvVideoCaptureExit(PyObject* self, PyObject* /*args*/, PyObject* /*kw*/) {
    Ptr<cv::VideoCapture>* obj_getp = nullptr;
    if (pyopencv_VideoCapture_getp(self, obj_getp) && obj_getp && *obj_getp) {
        (*obj_getp)->release();
    }
    Py_RETURN_NONE;
}

static PyObject* pycvVideoWriterExit(PyObject* self, PyObject* /*args*/, PyObject* /*kw*/) {
    Ptr<cv::VideoWriter>* obj_getp = nullptr;
    if (pyopencv_VideoWriter_getp(self, obj_getp) && obj_getp && *obj_getp) {
        (*obj_getp)->release();
    }
    Py_RETURN_NONE;
}

#define PYOPENCV_EXTRA_METHODS_VideoCapture \
    {"__enter__", CV_PY_FN_WITH_KW(pycvVideoEnter), "Context manager enter"}, \
    {"__exit__",  CV_PY_FN_WITH_KW(pycvVideoCaptureExit), "Context manager exit"},

#define PYOPENCV_EXTRA_METHODS_VideoWriter \
    {"__enter__", CV_PY_FN_WITH_KW(pycvVideoEnter), "Context manager enter"}, \
    {"__exit__",  CV_PY_FN_WITH_KW(pycvVideoWriterExit), "Context manager exit"},


#endif // HAVE_OPENCV_VIDEOIO
