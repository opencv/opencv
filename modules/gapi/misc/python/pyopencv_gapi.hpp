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
