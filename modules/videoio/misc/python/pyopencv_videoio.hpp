#ifdef HAVE_OPENCV_VIDEOIO
#include "../../src/cap_interface.hpp"
typedef std::vector<VideoCaptureAPIs> vector_VideoCaptureAPIs;
typedef std::vector<VideoCapture> vector_VideoCapture;
typedef CvStream streambuf;

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

static long long readIO(void* opaque, char* buffer, long long n)
{
    PyObject* ioBase = reinterpret_cast<PyObject*>(opaque);

    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    PyObject* size = pyopencv_from(static_cast<int>(n));

    PyObject* res = PyObject_CallMethodObjArgs(ioBase, PyString_FromString("read"), size, NULL);
    char* src = PyBytes_AsString(res);
    size_t len = static_cast<size_t>(PyBytes_Size(res));
    std::memcpy(buffer, src, len);
    Py_DECREF(res);
    Py_DECREF(size);

    PyGILState_Release(gstate);

    return len;
}

static long long seekIO(void* opaque, long long offset, int way)
{
    PyObject* ioBase = reinterpret_cast<PyObject*>(opaque);

    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    PyObject* size = pyopencv_from(static_cast<int>(offset));
    PyObject* whence = pyopencv_from(way);

    PyObject* res = PyObject_CallMethodObjArgs(ioBase, PyString_FromString("seek"), size, whence, NULL);
    int pos = PyLong_AsLong(res);
    Py_DECREF(res);
    Py_DECREF(size);
    Py_DECREF(whence);

    PyGILState_Release(gstate);

    return pos;
}

template<>
bool pyopencv_to(PyObject* obj, CvStream& p, const ArgInfo&)
{
    if (!obj)
        return false;

    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    PyObject* ioModule = PyImport_ImportModule("io");
    PyObject* type = PyObject_GetAttrString(ioModule, "BufferedIOBase");
    Py_DECREF(ioModule);
    if (!PyObject_IsInstance(obj, type))
        CV_Error(cv::Error::StsBadArg, "Input stream should be derived from io.BufferedIOBase");
    Py_DECREF(type);
    PyGILState_Release(gstate);

    p = CvStream(obj, readIO, seekIO);
    return true;
}

#endif // HAVE_OPENCV_VIDEOIO
