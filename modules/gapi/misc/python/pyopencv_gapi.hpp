#ifndef OPENCV_GAPI_PYOPENCV_GAPI_HPP
#define OPENCV_GAPI_PYOPENCV_GAPI_HPP

#ifdef HAVE_OPENCV_GAPI

// NB: Python wrapper replaces :: with _ for classes
using gapi_GKernelPackage = cv::gapi::GKernelPackage;
using gapi_GNetPackage = cv::gapi::GNetPackage;
using gapi_ie_PyParams = cv::gapi::ie::PyParams;
using gapi_wip_IStreamSource_Ptr = cv::Ptr<cv::gapi::wip::IStreamSource>;
using detail_ExtractArgsCallback = cv::detail::ExtractArgsCallback;
using detail_ExtractMetaCallback = cv::detail::ExtractMetaCallback;

// FIXME: Python wrapper generate code without namespace std,
// so it cause error: "string wasn't declared"
// WA: Create using
using std::string;

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
        case GRunArg::index_of<cv::detail::VectorRef>():
        {
            const auto& vref = util::get<cv::detail::VectorRef>(v);
            switch (vref.getKind())
            {
                case cv::detail::OpaqueKind::CV_POINT2F:
                    return pyopencv_from(vref.rref<cv::Point2f>());
                default:
                    PyErr_SetString(PyExc_TypeError, "Unsupported kind for GArray");
                    return NULL;
            }
        }
        default:
            PyErr_SetString(PyExc_TypeError, "Failed to unpack GRunArgs");
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

template<>
bool pyopencv_to(PyObject* obj, GMetaArgs& value, const ArgInfo& info)
{
    return pyopencv_to_generic_vec(obj, value, info);
}

template<>
PyObject* pyopencv_from(const GMetaArgs& value)
{
    return pyopencv_from_generic_vec(value);
}

template <typename T>
void pyopencv_to_with_check(PyObject* from, T& to, const std::string& msg = "")
{
    if (!pyopencv_to(from, to, ArgInfo("", false)))
    {
        cv::util::throw_error(std::logic_error(msg));
    }
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
        // FIXME: Deprecated. Will be removed in next PRs
        else if (PyObject_TypeCheck(item, reinterpret_cast<PyTypeObject*>(pyopencv_GArrayP2f_TypePtr)))
        {
            args.emplace_back(reinterpret_cast<pyopencv_GArrayP2f_t*>(item)->v.strip());
        }
        else
        {
            PyErr_SetString(PyExc_TypeError, "Unsupported type for cv.GIn()/cv.GOut()");
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

static cv::GRunArg extract_run_arg(const cv::GTypeInfo& info, PyObject* item)
{
    switch (info.shape)
    {
        case cv::GShape::GMAT:
        {
            // NB: In case streaming it can be IStreamSource or cv::Mat
            if (PyObject_TypeCheck(item,
                        reinterpret_cast<PyTypeObject*>(pyopencv_gapi_wip_IStreamSource_TypePtr)))
            {
                cv::gapi::wip::IStreamSource::Ptr source =
                    reinterpret_cast<pyopencv_gapi_wip_IStreamSource_t*>(item)->v;
                return source;
            }
            else
            {
                cv::Mat obj;
                pyopencv_to_with_check(item, obj, "Failed to obtain cv::Mat");
                return obj;
            }
        }
        case cv::GShape::GSCALAR:
        {
            cv::Scalar obj;
            pyopencv_to_with_check(item, obj, "Failed to obtain cv::Scalar");
            return obj;
        }
        default:
            util::throw_error(std::logic_error("Unsupported output shape"));
    }
    GAPI_Assert(false && "Unreachable code");
}

static cv::GRunArgs extract_run_args(const cv::GTypesInfo& info, PyObject* py_args)
{
    cv::GRunArgs args;
    Py_ssize_t tuple_size = PyTuple_Size(py_args);
    args.reserve(tuple_size);

    for (int i = 0; i < tuple_size; ++i)
    {
        args.push_back(extract_run_arg(info[i], PyTuple_GetItem(py_args, i)));
    }

    return args;
}

static cv::GMetaArg extract_meta_arg(const cv::GTypeInfo& info, PyObject* item)
{
    switch (info.shape)
    {
        case cv::GShape::GMAT:
        {
            cv::Mat obj;
            pyopencv_to_with_check(item, obj, "Failed to obtain cv::Mat");
            return cv::GMetaArg{cv::descr_of(obj)};
        }
        case cv::GShape::GSCALAR:
        {
            cv::Scalar obj;
            pyopencv_to_with_check(item, obj, "Failed to obtain cv::Scalar");
            return cv::GMetaArg{cv::descr_of(obj)};
        }
        default:
            util::throw_error(std::logic_error("Unsupported output shape"));
    }
}

static cv::GMetaArgs extract_meta_args(const cv::GTypesInfo& info, PyObject* py_args)
{
    cv::GMetaArgs metas;
    Py_ssize_t tuple_size = PyTuple_Size(py_args);
    metas.reserve(tuple_size);

    for (int i = 0; i < tuple_size; ++i)
    {
        metas.push_back(extract_meta_arg(info[i], PyTuple_GetItem(py_args, i)));
    }

    return metas;
}

static PyObject* pyopencv_cv_gin(PyObject*, PyObject* py_args)
{
    Py_INCREF(py_args);
    auto callback = cv::detail::ExtractArgsCallback{[=](const cv::GTypesInfo& info)
        {
            PyGILState_STATE gstate;
            gstate = PyGILState_Ensure();

            cv::GRunArgs args;
            try
            {
                args = extract_run_args(info, py_args);
            }
            catch (...)
            {
                PyGILState_Release(gstate);
                throw;
            }
            PyGILState_Release(gstate);
            return args;
        }};

    return pyopencv_from(callback);
}

static PyObject* pyopencv_cv_descr_of(PyObject*, PyObject* py_args)
{
    Py_INCREF(py_args);
    auto callback = cv::detail::ExtractMetaCallback{[=](const cv::GTypesInfo& info)
        {
            PyGILState_STATE gstate;
            gstate = PyGILState_Ensure();

            cv::GMetaArgs args;
            try
            {
                args = extract_meta_args(info, py_args);
            }
            catch (...)
            {
                PyGILState_Release(gstate);
                throw;
            }
            PyGILState_Release(gstate);
            return args;
        }};
    return pyopencv_from(callback);
}

#endif  // HAVE_OPENCV_GAPI
#endif  // OPENCV_GAPI_PYOPENCV_GAPI_HPP
