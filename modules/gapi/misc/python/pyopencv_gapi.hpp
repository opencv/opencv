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

template <typename T>
static PyObject* extract_proto_args(PyObject* py_args, PyObject* kw)
{
    using namespace cv;

    GProtoArgs args;
    Py_ssize_t size = PyTuple_Size(py_args);
    for (int i = 0; i < size; ++i) {
        PyObject* item = PyTuple_GetItem(py_args, i);
        if (PyObject_TypeCheck(item, reinterpret_cast<PyTypeObject*>(pyopencv_GScalar_TypePtr))) {
            args.emplace_back(reinterpret_cast<pyopencv_GScalar_t*>(item)->v);
        } else if (PyObject_TypeCheck(item, reinterpret_cast<PyTypeObject*>(pyopencv_GMat_TypePtr))) {
            args.emplace_back(reinterpret_cast<pyopencv_GMat_t*>(item)->v);
        } else {
            PyErr_SetString(PyExc_TypeError, "cv.GIn() supports only cv.GMat and cv.GScalar");
            return NULL;
        }
    }

    return pyopencv_from<T>(T{std::move(args)});
}

static PyObject* pyopencv_cv_GIn(PyObject* , PyObject* py_args, PyObject* kw)
{
    return extract_proto_args<GProtoInputArgs>(py_args, kw);
}

static PyObject* pyopencv_cv_GOut(PyObject* , PyObject* py_args, PyObject* kw)
{
    return extract_proto_args<GProtoOutputArgs>(py_args, kw);
}
