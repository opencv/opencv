#ifndef OPENCV_GAPI_PYOPENCV_GAPI_HPP
#define OPENCV_GAPI_PYOPENCV_GAPI_HPP

#ifdef HAVE_OPENCV_GAPI

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

template<>
bool pyopencv_to(PyObject* obj, GRunArgs& value, const ArgInfo& info)
{
    return pyopencv_to_generic_vec(obj, value, info);
}

static PyObject* from_grunarg(const GRunArg& v)
{
    switch (v.index())
    {
        case GRunArg::index_of<cv::Mat>():
        {
            const auto& m = util::get<cv::Mat>(v);
            return pyopencv_from(m);
        }

        case GRunArg::index_of<cv::Scalar>():
        {
            const auto& s = util::get<cv::Scalar>(v);
            return pyopencv_from(s);
        }

        default:
            return NULL;
    }
    GAPI_Assert(false);
}

template<>
PyObject* pyopencv_from(const GRunArgs& value)
{
    size_t i, n = value.size();

    // NB: It doesn't make sense to return list with a single element
    if (n == 1)
    {
        PyObject* item = from_grunarg(value[0]);
        if(!item)
        {
            PyErr_SetString(PyExc_TypeError, "Failed to unpack GRunArgs");
            return NULL;
        }
        return item;
    }

    PyObject* list = PyList_New(n);
    for(i = 0; i < n; ++i)
    {
        PyObject* item = from_grunarg(value[i]);
        if(!item)
        {
            Py_DECREF(list);
            PyErr_SetString(PyExc_TypeError, "Failed to unpack GRunArgs");
            return NULL;
        }
        PyList_SetItem(list, i, item);
    }

    return list;
}

template <typename T>
static PyObject* extract_proto_args(PyObject* py_args, PyObject* kw)
{
    using namespace cv;

    GProtoArgs args;
    Py_ssize_t size = PyTuple_Size(py_args);
    for (int i = 0; i < size; ++i)
    {
        PyObject* item = PyTuple_GetItem(py_args, i);
        if (PyObject_TypeCheck(item, reinterpret_cast<PyTypeObject*>(pyopencv_GScalar_TypePtr)))
        {
            args.emplace_back(reinterpret_cast<pyopencv_GScalar_t*>(item)->v);
        }
        else if (PyObject_TypeCheck(item, reinterpret_cast<PyTypeObject*>(pyopencv_GMat_TypePtr)))
        {
            args.emplace_back(reinterpret_cast<pyopencv_GMat_t*>(item)->v);
        }
        else
        {
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

static PyObject* pyopencv_cv_gin(PyObject* , PyObject* py_args, PyObject* kw)
{
    using namespace cv;

    GRunArgs args;
    Py_ssize_t size = PyTuple_Size(py_args);
    for (int i = 0; i < size; ++i)
    {
        PyObject* item = PyTuple_GetItem(py_args, i);
        if (PyTuple_Check(item))
        {
            cv::Scalar s;
            if (pyopencv_to(item, s, ArgInfo("scalar", false)))
            {
                args.emplace_back(s);
            }
            else
            {
                PyErr_SetString(PyExc_TypeError, "Failed convert tuple to cv::Scalar");
                return NULL;
            }
        }
        else if (PyArray_Check(item))
        {
            cv::Mat m;
            if (pyopencv_to(item, m, ArgInfo("mat", false)))
            {
                args.emplace_back(m);
            }
            else
            {
                PyErr_SetString(PyExc_TypeError, "Failed convert array to cv::Mat");
                return NULL;
            }
        }
    }

    return pyopencv_from_generic_vec(args);
}

static PyObject* pyopencv_cv_gout(PyObject* o, PyObject* py_args, PyObject* kw)
{
    return pyopencv_cv_gin(o, py_args, kw);
}

#endif  // HAVE_OPENCV_GAPI
#endif  // OPENCV_GAPI_PYOPENCV_GAPI_HPP
