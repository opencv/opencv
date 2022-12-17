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

#endif // HAVE_OPENCV_VIDEOIO
