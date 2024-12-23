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

class IOBaseWrapper : public std::streambuf
{
public:
    IOBaseWrapper(PyObject* _obj = nullptr) : obj(_obj)
    {
        if (obj)
            Py_INCREF(obj);
    }

    ~IOBaseWrapper()
    {
        if (obj)
            Py_DECREF(obj);
    }

    std::streamsize xsgetn(char* buf, std::streamsize n) override
    {
        PyObject* ioBase = reinterpret_cast<PyObject*>(obj);

        PyGILState_STATE gstate;
        gstate = PyGILState_Ensure();

        PyObject* size = pyopencv_from(static_cast<int>(n));

        PyObject* res = PyObject_CallMethodObjArgs(ioBase, PyString_FromString("read"), size, NULL);
        char* src = PyBytes_AsString(res);
        size_t len = static_cast<size_t>(PyBytes_Size(res));
        CV_CheckLE(len, static_cast<size_t>(n), "Stream chunk size should be less or equal than requested size");
        std::memcpy(buf, src, len);
        Py_DECREF(res);
        Py_DECREF(size);

        PyGILState_Release(gstate);

        return len;
    }

    std::streampos seekoff(std::streamoff off, std::ios_base::seekdir way, std::ios_base::openmode = std::ios_base::in | std::ios_base::out) override
    {
        PyObject* ioBase = reinterpret_cast<PyObject*>(obj);

        PyGILState_STATE gstate;
        gstate = PyGILState_Ensure();

        PyObject* size = pyopencv_from(static_cast<int>(off));
        PyObject* whence = pyopencv_from(way == std::ios_base::beg ? SEEK_SET : (way == std::ios_base::end ? SEEK_END : SEEK_CUR));

        PyObject* res = PyObject_CallMethodObjArgs(ioBase, PyString_FromString("seek"), size, whence, NULL);
        int pos = PyLong_AsLong(res);
        Py_DECREF(res);
        Py_DECREF(size);
        Py_DECREF(whence);

        PyGILState_Release(gstate);

        return pos;
    }

private:
    PyObject* obj;
};

template<>
bool pyopencv_to(PyObject* obj, Ptr<std::streambuf>& p, const ArgInfo&)
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

    p = makePtr<IOBaseWrapper>(obj);
    return true;
}

#endif // HAVE_OPENCV_VIDEOIO
