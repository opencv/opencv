#ifdef HAVE_OPENCV_VIDEOIO
typedef std::vector<VideoCaptureAPIs> vector_VideoCaptureAPIs;

template<>
bool pyopencv_to(PyObject *o, cv::VideoCaptureAPIs &v, const char *name)
{
    (void)name;
    v = CAP_ANY;
    if (!o || o == Py_None)
        return false;
    else if (PyLong_Check(o))
    {
        v = VideoCaptureAPIs((int64)PyLong_AsLongLong(o));
        return true;
    }
    else if (PyInt_Check(o))
    {
        v = VideoCaptureAPIs((int64)PyInt_AS_LONG(o));
        return true;
    }
    else
        return false;
}

template<>
PyObject* pyopencv_from(const cv::VideoCaptureAPIs &v)
{
    return pyopencv_from((int)(v));
}

template<> struct pyopencvVecConverter<cv::VideoCaptureAPIs>
{
    static bool to(PyObject* obj, std::vector<cv::VideoCaptureAPIs>& value, const ArgInfo info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const std::vector<cv::VideoCaptureAPIs>& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

template<>
bool pyopencv_to(PyObject *o, std::vector<cv::VideoCaptureAPIs>& apis, const char *name)
{
  return pyopencvVecConverter<cv::VideoCaptureAPIs>::to(o, apis, ArgInfo(name, false));
}

#endif // HAVE_OPENCV_VIDEOIO
