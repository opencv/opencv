using gapi_GKernelPackage = cv::gapi::GKernelPackage;

template<>
bool pyopencv_to(PyObject* obj, std::vector<GCompileArg>& value, const ArgInfo& info)
{
    return pyopencv_to_generic_vec(obj, value, info);
}

template<>
PyObject* pyopencv_from(const std::vector<GCompileArg>& value)
{
    return pyopencv_from_generic_vec(value);
}

static PyObject* pyopencv_cv_GIn(PyObject* , PyObject* py_args, PyObject* kw)
{
    using namespace cv;

    GProtoArgs args;
    GMat in1;
    Py_ssize_t size = PyTuple_Size(py_args);
    std::cout << "size = " << size << std::endl;
    for (int i = 0; i < size; ++i) {
        PyObject* item = PyTuple_GetItem(py_args, i);
        if (!pyopencv_to(item, in1, ArgInfo("in1", 0))) {
            std::cout << "FAILED caset " << std::endl;
            // args.emplace(in1);
        } else {
            std::cout << "SUCCESS CAST " << std::endl;
        }
    }
    return NULL;
}

template<>
struct PyOpenCV_Converter<GIOProtoArgs<In_Tag>>
{
    static PyObject* from(const GIOProtoArgs<In_Tag>& p)
    {
        return NULL;
    }

    static bool to(PyObject *o, GIOProtoArgs<In_Tag>& p, const ArgInfo& info)
    {
        return false;
    }
};

template<>
struct PyOpenCV_Converter<GIOProtoArgs<Out_Tag>>
{
    static PyObject* from(const GIOProtoArgs<Out_Tag>& p)
    {
        return NULL;
    }

    static bool to(PyObject *o, GIOProtoArgs<Out_Tag>& p, const ArgInfo& info)
    {
        return false;
    }
};
