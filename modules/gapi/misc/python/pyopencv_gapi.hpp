#ifndef OPENCV_GAPI_PYOPENCV_GAPI_HPP
#define OPENCV_GAPI_PYOPENCV_GAPI_HPP

#ifdef HAVE_OPENCV_GAPI

// NB: Python wrapper replaces :: with _ for classes
using gapi_GKernelPackage = cv::gapi::GKernelPackage;
using gapi_GNetPackage = cv::gapi::GNetPackage;
using gapi_ie_PyParams = cv::gapi::ie::PyParams;
using gapi_wip_IStreamSource_Ptr = cv::Ptr<cv::gapi::wip::IStreamSource>;

// NB: Python wrapper generate T_U for T<U>
// This behavior is only observed for inputs
using GOpaque_bool     = cv::GOpaque<bool>;
using GOpaque_int      = cv::GOpaque<int>;
using GOpaque_double   = cv::GOpaque<double>;
using GOpaque_float    = cv::GOpaque<double>;
using GOpaque_string   = cv::GOpaque<std::string>;
using GOpaque_Point    = cv::GOpaque<cv::Point>;
using GOpaque_Point2f  = cv::GOpaque<cv::Point2f>;
using GOpaque_Size     = cv::GOpaque<cv::Size>;
using GOpaque_Rect     = cv::GOpaque<cv::Rect>;

using GArray_bool     = cv::GArray<bool>;
using GArray_int      = cv::GArray<int>;
using GArray_double   = cv::GArray<double>;
using GArray_float    = cv::GArray<double>;
using GArray_string   = cv::GArray<std::string>;
using GArray_Point    = cv::GArray<cv::Point>;
using GArray_Point2f  = cv::GArray<cv::Point2f>;
using GArray_Size     = cv::GArray<cv::Size>;
using GArray_Rect     = cv::GArray<cv::Rect>;
using GArray_Scalar   = cv::GArray<cv::Scalar>;
using GArray_Mat      = cv::GArray<cv::Mat>;

// FIXME: Python wrapper generate code without namespace std,
// so it cause error: "string wasn't declared"
// WA: Create using
using std::string;

template <>
bool pyopencv_to(PyObject* obj, std::vector<GCompileArg>& value, const ArgInfo& info)
{
    return pyopencv_to_generic_vec(obj, value, info);
}

template <>
PyObject* pyopencv_from(const std::vector<GCompileArg>& value)
{
    return pyopencv_from_generic_vec(value);
}

template <>
bool pyopencv_to(PyObject* obj, GRunArgs& value, const ArgInfo& info)
{
    return pyopencv_to_generic_vec(obj, value, info);
}

template <>
PyObject* pyopencv_from(const cv::detail::OpaqueRef& oref)
{
    switch (oref.getKind())
    {
        case cv::detail::OpaqueKind::CV_BOOL    : return pyopencv_from(oref.rref<bool>());
        case cv::detail::OpaqueKind::CV_INT     : return pyopencv_from(oref.rref<int>());
        case cv::detail::OpaqueKind::CV_DOUBLE  : return pyopencv_from(oref.rref<double>());
        case cv::detail::OpaqueKind::CV_FLOAT   : return pyopencv_from(oref.rref<float>());
        case cv::detail::OpaqueKind::CV_STRING  : return pyopencv_from(oref.rref<std::string>());
        case cv::detail::OpaqueKind::CV_POINT   : return pyopencv_from(oref.rref<cv::Point>());
        case cv::detail::OpaqueKind::CV_POINT2F : return pyopencv_from(oref.rref<cv::Point2f>());
        case cv::detail::OpaqueKind::CV_SIZE    : return pyopencv_from(oref.rref<cv::Size>());
        case cv::detail::OpaqueKind::CV_RECT    : return pyopencv_from(oref.rref<cv::Rect>());
        default:
            PyErr_SetString(PyExc_TypeError, "Unsupported GOpaque type");
            return NULL;
    }
};

template <>
PyObject* pyopencv_from(const cv::detail::VectorRef& vr)
{
    switch (vr.getKind())
    {
        case cv::detail::OpaqueKind::CV_BOOL    : return pyopencv_from_generic_vec(vr.rref<bool>());
        case cv::detail::OpaqueKind::CV_INT     : return pyopencv_from_generic_vec(vr.rref<int>());
        case cv::detail::OpaqueKind::CV_DOUBLE  : return pyopencv_from_generic_vec(vr.rref<double>());
        case cv::detail::OpaqueKind::CV_FLOAT   : return pyopencv_from_generic_vec(vr.rref<float>());
        case cv::detail::OpaqueKind::CV_STRING  : return pyopencv_from_generic_vec(vr.rref<std::string>());
        case cv::detail::OpaqueKind::CV_POINT   : return pyopencv_from_generic_vec(vr.rref<cv::Point>());
        case cv::detail::OpaqueKind::CV_POINT2F : return pyopencv_from_generic_vec(vr.rref<cv::Point2f>());
        case cv::detail::OpaqueKind::CV_SIZE    : return pyopencv_from_generic_vec(vr.rref<cv::Size>());
        case cv::detail::OpaqueKind::CV_RECT    : return pyopencv_from_generic_vec(vr.rref<cv::Rect>());
        case cv::detail::OpaqueKind::CV_SCALAR  : return pyopencv_from_generic_vec(vr.rref<cv::Scalar>());
        case cv::detail::OpaqueKind::CV_MAT     : return pyopencv_from_generic_vec(vr.rref<cv::Mat>());
        default:
            PyErr_SetString(PyExc_TypeError, "Unsupported GArray type");
            return NULL;
    }
    GAPI_Assert(false);
}

template <>
PyObject* pyopencv_from(const GRunArg& v)
{
    switch (v.index())
    {
        case GRunArg::index_of<cv::Mat>():
            return pyopencv_from(util::get<cv::Mat>(v));

        case GRunArg::index_of<cv::Scalar>():
            return pyopencv_from(util::get<cv::Scalar>(v));

        case GRunArg::index_of<cv::detail::VectorRef>():
            return pyopencv_from(util::get<cv::detail::VectorRef>(v));

        case GRunArg::index_of<cv::detail::OpaqueRef>():
            return pyopencv_from(util::get<cv::detail::OpaqueRef>(v));

        default:
            PyErr_SetString(PyExc_TypeError, "Failed to unpack GRunArgs");
            return NULL;
    }
    GAPI_Assert(false && "Unreachable code");
}

template<>
PyObject* pyopencv_from(const GRunArgs& value)
{
    size_t i, n = value.size();

    // NB: It doesn't make sense to return list with a single element
    if (n == 1)
    {
        PyObject* item = pyopencv_from(value[0]);
        if(!item)
        {
            return NULL;
        }
        return item;
    }

    PyObject* list = PyList_New(n);
    for(i = 0; i < n; ++i)
    {
        PyObject* item = pyopencv_from(value[i]);
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
static PyObject* extract_proto_args(PyObject* py_args, PyObject* kw)
{
    using namespace cv;

    GProtoArgs args;
    Py_ssize_t size = PyTuple_Size(py_args);
    args.reserve(size);
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
        else if (PyObject_TypeCheck(item, reinterpret_cast<PyTypeObject*>(pyopencv_GOpaqueT_TypePtr)))
        {
            args.emplace_back(reinterpret_cast<pyopencv_GOpaqueT_t*>(item)->v.strip());
        }
        else if (PyObject_TypeCheck(item, reinterpret_cast<PyTypeObject*>(pyopencv_GArrayT_TypePtr)))
        {
            args.emplace_back(reinterpret_cast<pyopencv_GArrayT_t*>(item)->v.strip());
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

template <typename T>
void pyopencv_to_with_check(PyObject* from, T& to, const std::string& msg = "")
{
    if (!pyopencv_to(from, to, ArgInfo("", false)))
    {
        cv::util::throw_error(std::logic_error(msg));
    }
}

template <typename T>
void pyopencv_to_generic_vec_with_check(PyObject* from,
                                        std::vector<T>& to,
                                        const std::string& msg = "")
{
    if (!pyopencv_to_generic_vec(from, to, ArgInfo("", false)))
    {
        cv::util::throw_error(std::logic_error(msg));
    }
}

static cv::detail::OpaqueRef extractOpaqueRef(PyObject* from, cv::detail::OpaqueKind kind)
{
#define HANDLE_CASE(T, O) case cv::detail::OpaqueKind::CV_##T:  \
{                                                               \
    O obj;                                                      \
    pyopencv_to_with_check(from, obj, "Failed to obtain " # O); \
    return cv::detail::OpaqueRef{std::move(obj)};               \
}
        switch (kind)
        {
            HANDLE_CASE(BOOL,    bool);
            HANDLE_CASE(INT,     int);
            HANDLE_CASE(DOUBLE,  double);
            HANDLE_CASE(FLOAT,   float);
            HANDLE_CASE(STRING,  std::string);
            HANDLE_CASE(POINT,   cv::Point);
            HANDLE_CASE(POINT2F, cv::Point2f);
            HANDLE_CASE(SIZE,    cv::Size);
            HANDLE_CASE(RECT,    cv::Rect);
#undef HANDLE_CASE
            default:
                util::throw_error(std::logic_error("Unsupported type for GOpaqueT"));
        }
        GAPI_Assert(false && "Unreachable code");
}

static cv::detail::VectorRef extractVectorRef(PyObject* from, cv::detail::OpaqueKind kind)
{
#define HANDLE_CASE(T, O) case cv::detail::OpaqueKind::CV_##T:                        \
{                                                                                     \
    std::vector<O> obj;                                                               \
    pyopencv_to_generic_vec_with_check(from, obj, "Failed to obtain vector of " # O); \
    return cv::detail::VectorRef{std::move(obj)};                                     \
}
        switch (kind)
        {
            HANDLE_CASE(BOOL,    bool);
            HANDLE_CASE(INT,     int);
            HANDLE_CASE(DOUBLE,  double);
            HANDLE_CASE(FLOAT,   float);
            HANDLE_CASE(STRING,  std::string);
            HANDLE_CASE(POINT,   cv::Point);
            HANDLE_CASE(POINT2F, cv::Point2f);
            HANDLE_CASE(SIZE,    cv::Size);
            HANDLE_CASE(RECT,    cv::Rect);
            HANDLE_CASE(SCALAR,  cv::Scalar);
            HANDLE_CASE(MAT,     cv::Mat);
#undef HANDLE_CASE
            default:
                util::throw_error(std::logic_error("Unsupported type for GOpaqueT"));
        }
        GAPI_Assert(false && "Unreachable code");
}

static cv::GRunArgs extractRunArgs(const cv::GTypesInfo& info, PyObject* py_args)
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    cv::GRunArgs args;
    Py_ssize_t tuple_size = PyTuple_Size(py_args);
    args.reserve(tuple_size);

    for (int i = 0; i < tuple_size; ++i)
    {
        PyObject* item = PyTuple_GetItem(py_args, i);
        switch (info[i].shape)
        {
            case cv::GShape::GMAT:
            {
                // NB: In case streaming it can be IStreamSource or cv::Mat
                if (PyObject_TypeCheck(item,
                            reinterpret_cast<PyTypeObject*>(pyopencv_gapi_wip_IStreamSource_TypePtr)))
                {
                    cv::gapi::wip::IStreamSource::Ptr source =
                        reinterpret_cast<pyopencv_gapi_wip_IStreamSource_t*>(item)->v;
                    args.emplace_back(source);
                }
                else
                {
                    cv::Mat obj;
                    pyopencv_to_with_check(item, obj, "Failed to obtain cv::Mat");
                    args.emplace_back(obj);
                }
                break;
            }
            case cv::GShape::GSCALAR:
            {
                cv::Scalar obj;
                pyopencv_to_with_check(item, obj, "Failed to obtain cv::Scalar");
                args.emplace_back(obj);
                break;
            }
            case cv::GShape::GOPAQUE:
            {
                args.emplace_back(extractOpaqueRef(item, info[i].kind));
                break;
            }
            case cv::GShape::GARRAY:
            {
                args.emplace_back(extractVectorRef(item, info[i].kind));
                break;
            }
            default:
                util::throw_error(std::logic_error("Unsupported output shape"));
        }
    }
    PyGILState_Release(gstate);
    return args;
}

static PyObject* pyopencv_cv_gin(PyObject*, PyObject* py_args)
{
    Py_INCREF(py_args);
    cv::ExtractArgsCallback callback = std::bind(extractRunArgs, std::placeholders::_1, py_args);
    return pyopencv_from(callback);
}

template<typename T>
struct PyOpenCV_Converter<cv::GArray<T>>
{
    static PyObject* from(const cv::GArray<T>& p)
    {
        return pyopencv_from(cv::GArrayT(p));
    }
    static bool to(PyObject *obj, cv::GArray<T>& value, const ArgInfo& info)
    {
        if (PyObject_TypeCheck(obj, reinterpret_cast<PyTypeObject*>(pyopencv_GArrayT_TypePtr)))
        {
            auto& array = reinterpret_cast<pyopencv_GArrayT_t*>(obj)->v;
            try {
                value = cv::util::get<cv::GArray<T>>(array.arg);
            } catch (...) {
                return false;
            }
            return true;
        }
        return false;
    }
};

template<typename T>
struct PyOpenCV_Converter<cv::GOpaque<T>>
{
    static PyObject* from(const cv::GOpaque<T>& p)
    {
        return pyopencv_from(cv::GOpaqueT(p));
    }
    static bool to(PyObject *obj, cv::GOpaque<T>& value, const ArgInfo& info)
    {
        if (PyObject_TypeCheck(obj, reinterpret_cast<PyTypeObject*>(pyopencv_GOpaqueT_TypePtr)))
        {
            auto& opaque = reinterpret_cast<pyopencv_GOpaqueT_t*>(obj)->v;
            try {
                value = cv::util::get<cv::GOpaque<T>>(opaque.arg);
            } catch (...) {
                return false;
            }
            return true;
        }
        return false;
    }
};

#endif // HAVE_OPENCV_GAPI
#endif // OPENCV_GAPI_PYOPENCV_GAPI_HPP
