//warning number '5033' not a valid compiler warning in vc12
#if defined(_MSC_VER) && (_MSC_VER > 1800)
// eliminating duplicated round() declaration
#define HAVE_ROUND 1
#pragma warning(push)
#pragma warning(disable:5033)  // 'register' is no longer a supported storage class
#endif

// #define CVPY_DYNAMIC_INIT
// #define Py_DEBUG

#if defined(CVPY_DYNAMIC_INIT) && !defined(Py_DEBUG)
#   define Py_LIMITED_API 0x03030000
#endif

#include <cmath>
#include <Python.h>
#include <limits>

#if PY_MAJOR_VERSION < 3
#undef CVPY_DYNAMIC_INIT
#else
#define CV_PYTHON_3 1
#endif

#if defined(_MSC_VER) && (_MSC_VER > 1800)
#pragma warning(pop)
#endif

#define MODULESTR "cv2"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include "opencv2/opencv_modules.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/utils/configuration.private.hpp"
#include "opencv2/core/utils/logger.hpp"
#include "opencv2/core/utils/tls.hpp"

#include "pyopencv_generated_include.h"
#include "opencv2/core/types_c.h"
#include "pycompat.hpp"
#include <map>

#define CV_HAS_CONVERSION_ERROR(x) (((x) == -1) && PyErr_Occurred())

static PyObject* opencv_error = NULL;

class ArgInfo
{
public:
    const char* name;
    bool outputarg;
    // more fields may be added if necessary

    ArgInfo(const char* name_, bool outputarg_) : name(name_), outputarg(outputarg_) {}

private:
    ArgInfo(const ArgInfo&); // = delete
    ArgInfo& operator=(const ArgInfo&); // = delete
};

template<typename T, class TEnable = void>  // TEnable is used for SFINAE checks
struct PyOpenCV_Converter
{
    //static inline bool to(PyObject* obj, T& p, const ArgInfo& info);
    //static inline PyObject* from(const T& src);
};

// exception-safe pyopencv_to
template<typename _Tp> static
bool pyopencv_to_safe(PyObject* obj, _Tp& value, const ArgInfo& info)
{
    try
    {
        return pyopencv_to(obj, value, info);
    }
    catch (const std::exception &e)
    {
        PyErr_SetString(opencv_error, cv::format("Conversion error: %s, what: %s", info.name, e.what()).c_str());
        return false;
    }
    catch (...)
    {
        PyErr_SetString(opencv_error, cv::format("Conversion error: %s", info.name).c_str());
        return false;
    }
}

template<typename T> static
bool pyopencv_to(PyObject* obj, T& p, const ArgInfo& info) { return PyOpenCV_Converter<T>::to(obj, p, info); }

template<typename T> static
PyObject* pyopencv_from(const T& src) { return PyOpenCV_Converter<T>::from(src); }

static bool isPythonBindingsDebugEnabled()
{
    static bool param_debug = cv::utils::getConfigurationParameterBool("OPENCV_PYTHON_DEBUG", false);
    return param_debug;
}

static void emit_failmsg(PyObject * exc, const char *msg)
{
    static bool param_debug = isPythonBindingsDebugEnabled();
    if (param_debug)
    {
        CV_LOG_WARNING(NULL, "Bindings conversion failed: " << msg);
    }
    PyErr_SetString(exc, msg);
}

static int failmsg(const char *fmt, ...)
{
    char str[1000];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(str, sizeof(str), fmt, ap);
    va_end(ap);

    emit_failmsg(PyExc_TypeError, str);
    return 0;
}

static PyObject* failmsgp(const char *fmt, ...)
{
    char str[1000];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(str, sizeof(str), fmt, ap);
    va_end(ap);

    emit_failmsg(PyExc_TypeError, str);
    return 0;
}

class PyAllowThreads
{
public:
    PyAllowThreads() : _state(PyEval_SaveThread()) {}
    ~PyAllowThreads()
    {
        PyEval_RestoreThread(_state);
    }
private:
    PyThreadState* _state;
};

class PyEnsureGIL
{
public:
    PyEnsureGIL() : _state(PyGILState_Ensure()) {}
    ~PyEnsureGIL()
    {
        PyGILState_Release(_state);
    }
private:
    PyGILState_STATE _state;
};

/**
 * Light weight RAII wrapper for `PyObject*` owning references.
 * In comparisson to C++11 `std::unique_ptr` with custom deleter, it provides
 * implicit conversion functions that might be useful to initialize it with
 * Python functions those returns owning references through the `PyObject**`
 * e.g. `PyErr_Fetch` or directly pass it to functions those want to borrow
 * reference to object (doesn't extend object lifetime) e.g. `PyObject_Str`.
 */
class PySafeObject
{
public:
    PySafeObject() : obj_(NULL) {}

    explicit PySafeObject(PyObject* obj) : obj_(obj) {}

    ~PySafeObject()
    {
        Py_CLEAR(obj_);
    }

    operator PyObject*()
    {
        return obj_;
    }

    operator PyObject**()
    {
        return &obj_;
    }

    PyObject* release()
    {
        PyObject* obj = obj_;
        obj_ = NULL;
        return obj;
    }

private:
    PyObject* obj_;

    // Explicitly disable copy operations
    PySafeObject(const PySafeObject*); // = delete
    PySafeObject& operator=(const PySafeObject&); // = delete
};

static void pyRaiseCVException(const cv::Exception &e)
{
    PyObject_SetAttrString(opencv_error, "file", PyString_FromString(e.file.c_str()));
    PyObject_SetAttrString(opencv_error, "func", PyString_FromString(e.func.c_str()));
    PyObject_SetAttrString(opencv_error, "line", PyInt_FromLong(e.line));
    PyObject_SetAttrString(opencv_error, "code", PyInt_FromLong(e.code));
    PyObject_SetAttrString(opencv_error, "msg", PyString_FromString(e.msg.c_str()));
    PyObject_SetAttrString(opencv_error, "err", PyString_FromString(e.err.c_str()));
    PyErr_SetString(opencv_error, e.what());
}

#define ERRWRAP2(expr) \
try \
{ \
    PyAllowThreads allowThreads; \
    expr; \
} \
catch (const cv::Exception &e) \
{ \
    pyRaiseCVException(e); \
    return 0; \
} \
catch (const std::exception &e) \
{ \
    PyErr_SetString(opencv_error, e.what()); \
    return 0; \
} \
catch (...) \
{ \
    PyErr_SetString(opencv_error, "Unknown C++ exception from OpenCV code"); \
    return 0; \
}

using namespace cv;


namespace {
template<class T>
NPY_TYPES asNumpyType()
{
    return NPY_OBJECT;
}

template<>
NPY_TYPES asNumpyType<bool>()
{
    return NPY_BOOL;
}

#define CV_GENERATE_INTEGRAL_TYPE_NPY_CONVERSION(src, dst) \
    template<>                                             \
    NPY_TYPES asNumpyType<src>()                           \
    {                                                      \
        return NPY_##dst;                                  \
    }                                                      \
    template<>                                             \
    NPY_TYPES asNumpyType<u##src>()                        \
    {                                                      \
        return NPY_U##dst;                                 \
    }

CV_GENERATE_INTEGRAL_TYPE_NPY_CONVERSION(int8_t, INT8);

CV_GENERATE_INTEGRAL_TYPE_NPY_CONVERSION(int16_t, INT16);

CV_GENERATE_INTEGRAL_TYPE_NPY_CONVERSION(int32_t, INT32);

CV_GENERATE_INTEGRAL_TYPE_NPY_CONVERSION(int64_t, INT64);

#undef CV_GENERATE_INTEGRAL_TYPE_NPY_CONVERSION

template<>
NPY_TYPES asNumpyType<float>()
{
    return NPY_FLOAT;
}

template<>
NPY_TYPES asNumpyType<double>()
{
    return NPY_DOUBLE;
}

template <class T>
PyArray_Descr* getNumpyTypeDescriptor()
{
    return PyArray_DescrFromType(asNumpyType<T>());
}

template <>
PyArray_Descr* getNumpyTypeDescriptor<size_t>()
{
#if SIZE_MAX == ULONG_MAX
    return PyArray_DescrFromType(NPY_ULONG);
#elif SIZE_MAX == ULLONG_MAX
    return PyArray_DescrFromType(NPY_ULONGLONG);
#else
    return PyArray_DescrFromType(NPY_UINT);
#endif
}

template <class T, class U>
bool isRepresentable(U value) {
    return (std::numeric_limits<T>::min() <= value) && (value <= std::numeric_limits<T>::max());
}

template<class T>
bool canBeSafelyCasted(PyObject* obj, PyArray_Descr* to)
{
    return PyArray_CanCastTo(PyArray_DescrFromScalar(obj), to) != 0;
}


template<>
bool canBeSafelyCasted<size_t>(PyObject* obj, PyArray_Descr* to)
{
    PyArray_Descr* from = PyArray_DescrFromScalar(obj);
    if (PyArray_CanCastTo(from, to))
    {
        return true;
    }
    else
    {
        // False negative scenarios:
        // - Signed input is positive so it can be safely cast to unsigned output
        // - Input has wider limits but value is representable within output limits
        // - All the above
        if (PyDataType_ISSIGNED(from))
        {
            int64_t input = 0;
            PyArray_CastScalarToCtype(obj, &input, getNumpyTypeDescriptor<int64_t>());
            return (input >= 0) && isRepresentable<size_t>(static_cast<uint64_t>(input));
        }
        else
        {
            uint64_t input = 0;
            PyArray_CastScalarToCtype(obj, &input, getNumpyTypeDescriptor<uint64_t>());
            return isRepresentable<size_t>(input);
        }
        return false;
    }
}


template<class T>
bool parseNumpyScalar(PyObject* obj, T& value)
{
    if (PyArray_CheckScalar(obj))
    {
        // According to the numpy documentation:
        // There are 21 statically-defined PyArray_Descr objects for the built-in data-types
        // So descriptor pointer is not owning.
        PyArray_Descr* to = getNumpyTypeDescriptor<T>();
        if (canBeSafelyCasted<T>(obj, to))
        {
            PyArray_CastScalarToCtype(obj, &value, to);
            return true;
        }
    }
    return false;
}

TLSData<std::vector<std::string> > conversionErrorsTLS;

inline void pyPrepareArgumentConversionErrorsStorage(std::size_t size)
{
    std::vector<std::string>& conversionErrors = conversionErrorsTLS.getRef();
    conversionErrors.clear();
    conversionErrors.reserve(size);
}

void pyRaiseCVOverloadException(const std::string& functionName)
{
    const std::vector<std::string>& conversionErrors = conversionErrorsTLS.getRef();
    const std::size_t conversionErrorsCount = conversionErrors.size();
    if (conversionErrorsCount > 0)
    {
        // In modern std libraries small string optimization is used = no dynamic memory allocations,
        // but it can be applied only for string with length < 18 symbols (in GCC)
        const std::string bullet = "\n - ";

        // Estimate required buffer size - save dynamic memory allocations = faster
        std::size_t requiredBufferSize = bullet.size() * conversionErrorsCount;
        for (std::size_t i = 0; i < conversionErrorsCount; ++i)
        {
            requiredBufferSize += conversionErrors[i].size();
        }

        // Only string concatenation is required so std::string is way faster than
        // std::ostringstream
        std::string errorMessage("Overload resolution failed:");
        errorMessage.reserve(errorMessage.size() + requiredBufferSize);
        for (std::size_t i = 0; i < conversionErrorsCount; ++i)
        {
            errorMessage += bullet;
            errorMessage += conversionErrors[i];
        }
        cv::Exception exception(CV_StsBadArg, errorMessage, functionName, "", -1);
        pyRaiseCVException(exception);
    }
    else
    {
        cv::Exception exception(CV_StsInternal, "Overload resolution failed, but no errors reported",
                                functionName, "", -1);
        pyRaiseCVException(exception);
    }
}

void pyPopulateArgumentConversionErrors()
{
    if (PyErr_Occurred())
    {
        PySafeObject exception_type;
        PySafeObject exception_value;
        PySafeObject exception_traceback;
        PyErr_Fetch(exception_type, exception_value, exception_traceback);
        PyErr_NormalizeException(exception_type, exception_value,
                                 exception_traceback);

        PySafeObject exception_message(PyObject_Str(exception_value));
        std::string message;
        getUnicodeString(exception_message, message);
#ifdef CV_CXX11
        conversionErrorsTLS.getRef().push_back(std::move(message));
#else
        conversionErrorsTLS.getRef().push_back(message);
#endif
    }
}

struct SafeSeqItem
{
    PyObject * item;
    SafeSeqItem(PyObject *obj, size_t idx) { item = PySequence_GetItem(obj, idx); }
    ~SafeSeqItem() { Py_XDECREF(item); }

private:
    SafeSeqItem(const SafeSeqItem&); // = delete
    SafeSeqItem& operator=(const SafeSeqItem&); // = delete
};

template <class T>
class RefWrapper
{
public:
    RefWrapper(T& item) : item_(item) {}

    T& get() CV_NOEXCEPT { return item_; }

private:
    T& item_;
};

// In order to support this conversion on 3.x branch - use custom reference_wrapper
// and C-style array instead of std::array<T, N>
template <class T, std::size_t N>
bool parseSequence(PyObject* obj, RefWrapper<T> (&value)[N], const ArgInfo& info)
{
    if (!obj || obj == Py_None)
    {
        return true;
    }
    if (!PySequence_Check(obj))
    {
        failmsg("Can't parse '%s'. Input argument doesn't provide sequence "
                "protocol", info.name);
        return false;
    }
    const std::size_t sequenceSize = PySequence_Size(obj);
    if (sequenceSize != N)
    {
        failmsg("Can't parse '%s'. Expected sequence length %lu, got %lu",
                info.name, N, sequenceSize);
        return false;
    }
    for (std::size_t i = 0; i < N; ++i)
    {
        SafeSeqItem seqItem(obj, i);
        if (!pyopencv_to(seqItem.item, value[i].get(), info))
        {
            failmsg("Can't parse '%s'. Sequence item with index %lu has a "
                    "wrong type", info.name, i);
            return false;
        }
    }
    return true;
}
} // namespace

namespace traits {
template <bool Value>
struct BooleanConstant
{
    static const bool value = Value;
    typedef BooleanConstant<Value> type;
};

typedef BooleanConstant<true> TrueType;
typedef BooleanConstant<false> FalseType;

template <class T>
struct VoidType {
    typedef void type;
};

template <class T, class DType = void>
struct IsRepresentableAsMatDataType : FalseType
{
};

template <class T>
struct IsRepresentableAsMatDataType<T, typename VoidType<typename DataType<T>::channel_type>::type> : TrueType
{
};
} // namespace traits

typedef std::vector<uchar> vector_uchar;
typedef std::vector<char> vector_char;
typedef std::vector<int> vector_int;
typedef std::vector<float> vector_float;
typedef std::vector<double> vector_double;
typedef std::vector<size_t> vector_size_t;
typedef std::vector<Point> vector_Point;
typedef std::vector<Point2f> vector_Point2f;
typedef std::vector<Point3f> vector_Point3f;
typedef std::vector<Size> vector_Size;
typedef std::vector<Vec2f> vector_Vec2f;
typedef std::vector<Vec3f> vector_Vec3f;
typedef std::vector<Vec4f> vector_Vec4f;
typedef std::vector<Vec6f> vector_Vec6f;
typedef std::vector<Vec4i> vector_Vec4i;
typedef std::vector<Rect> vector_Rect;
typedef std::vector<Rect2d> vector_Rect2d;
typedef std::vector<RotatedRect> vector_RotatedRect;
typedef std::vector<KeyPoint> vector_KeyPoint;
typedef std::vector<Mat> vector_Mat;
typedef std::vector<std::vector<Mat> > vector_vector_Mat;
typedef std::vector<UMat> vector_UMat;
typedef std::vector<DMatch> vector_DMatch;
typedef std::vector<String> vector_String;
typedef std::vector<Scalar> vector_Scalar;

typedef std::vector<std::vector<char> > vector_vector_char;
typedef std::vector<std::vector<Point> > vector_vector_Point;
typedef std::vector<std::vector<Point2f> > vector_vector_Point2f;
typedef std::vector<std::vector<Point3f> > vector_vector_Point3f;
typedef std::vector<std::vector<DMatch> > vector_vector_DMatch;
typedef std::vector<std::vector<KeyPoint> > vector_vector_KeyPoint;

class NumpyAllocator : public MatAllocator
{
public:
    NumpyAllocator() { stdAllocator = Mat::getStdAllocator(); }
    ~NumpyAllocator() {}

    UMatData* allocate(PyObject* o, int dims, const int* sizes, int type, size_t* step) const
    {
        UMatData* u = new UMatData(this);
        u->data = u->origdata = (uchar*)PyArray_DATA((PyArrayObject*) o);
        npy_intp* _strides = PyArray_STRIDES((PyArrayObject*) o);
        for( int i = 0; i < dims - 1; i++ )
            step[i] = (size_t)_strides[i];
        step[dims-1] = CV_ELEM_SIZE(type);
        u->size = sizes[0]*step[0];
        u->userdata = o;
        return u;
    }

    UMatData* allocate(int dims0, const int* sizes, int type, void* data, size_t* step, int flags, UMatUsageFlags usageFlags) const CV_OVERRIDE
    {
        if( data != 0 )
        {
            // issue #6969: CV_Error(Error::StsAssert, "The data should normally be NULL!");
            // probably this is safe to do in such extreme case
            return stdAllocator->allocate(dims0, sizes, type, data, step, flags, usageFlags);
        }
        PyEnsureGIL gil;

        int depth = CV_MAT_DEPTH(type);
        int cn = CV_MAT_CN(type);
        const int f = (int)(sizeof(size_t)/8);
        int typenum = depth == CV_8U ? NPY_UBYTE : depth == CV_8S ? NPY_BYTE :
        depth == CV_16U ? NPY_USHORT : depth == CV_16S ? NPY_SHORT :
        depth == CV_32S ? NPY_INT : depth == CV_32F ? NPY_FLOAT :
        depth == CV_64F ? NPY_DOUBLE : f*NPY_ULONGLONG + (f^1)*NPY_UINT;
        int i, dims = dims0;
        cv::AutoBuffer<npy_intp> _sizes(dims + 1);
        for( i = 0; i < dims; i++ )
            _sizes[i] = sizes[i];
        if( cn > 1 )
            _sizes[dims++] = cn;
        PyObject* o = PyArray_SimpleNew(dims, _sizes.data(), typenum);
        if(!o)
            CV_Error_(Error::StsError, ("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims));
        return allocate(o, dims0, sizes, type, step);
    }

    bool allocate(UMatData* u, int accessFlags, UMatUsageFlags usageFlags) const CV_OVERRIDE
    {
        return stdAllocator->allocate(u, accessFlags, usageFlags);
    }

    void deallocate(UMatData* u) const CV_OVERRIDE
    {
        if(!u)
            return;
        PyEnsureGIL gil;
        CV_Assert(u->urefcount >= 0);
        CV_Assert(u->refcount >= 0);
        if(u->refcount == 0)
        {
            PyObject* o = (PyObject*)u->userdata;
            Py_XDECREF(o);
            delete u;
        }
    }

    const MatAllocator* stdAllocator;
};

NumpyAllocator g_numpyAllocator;


enum { ARG_NONE = 0, ARG_MAT = 1, ARG_SCALAR = 2 };

static bool isBool(PyObject* obj) CV_NOEXCEPT
{
    return PyArray_IsScalar(obj, Bool) || PyBool_Check(obj);
}

// special case, when the converter needs full ArgInfo structure
static bool pyopencv_to(PyObject* o, Mat& m, const ArgInfo& info)
{
    bool allowND = true;
    if(!o || o == Py_None)
    {
        if( !m.data )
            m.allocator = &g_numpyAllocator;
        return true;
    }

    if( PyInt_Check(o) )
    {
        double v[] = {static_cast<double>(PyInt_AsLong((PyObject*)o)), 0., 0., 0.};
        m = Mat(4, 1, CV_64F, v).clone();
        return true;
    }
    if( PyFloat_Check(o) )
    {
        double v[] = {PyFloat_AsDouble((PyObject*)o), 0., 0., 0.};
        m = Mat(4, 1, CV_64F, v).clone();
        return true;
    }
    if( PyTuple_Check(o) )
    {
        int i, sz = (int)PyTuple_Size((PyObject*)o);
        m = Mat(sz, 1, CV_64F);
        for( i = 0; i < sz; i++ )
        {
            PyObject* oi = PyTuple_GetItem(o, i);
            if( PyInt_Check(oi) )
                m.at<double>(i) = (double)PyInt_AsLong(oi);
            else if( PyFloat_Check(oi) )
                m.at<double>(i) = (double)PyFloat_AsDouble(oi);
            else
            {
                failmsg("%s is not a numerical tuple", info.name);
                m.release();
                return false;
            }
        }
        return true;
    }

    if( !PyArray_Check(o) )
    {
        failmsg("%s is not a numpy array, neither a scalar", info.name);
        return false;
    }

    PyArrayObject* oarr = (PyArrayObject*) o;

    bool needcopy = false, needcast = false;
    int typenum = PyArray_TYPE(oarr), new_typenum = typenum;
    int type = typenum == NPY_UBYTE ? CV_8U :
               typenum == NPY_BYTE ? CV_8S :
               typenum == NPY_USHORT ? CV_16U :
               typenum == NPY_SHORT ? CV_16S :
               typenum == NPY_INT ? CV_32S :
               typenum == NPY_INT32 ? CV_32S :
               typenum == NPY_FLOAT ? CV_32F :
               typenum == NPY_DOUBLE ? CV_64F : -1;

    if( type < 0 )
    {
        if( typenum == NPY_INT64 || typenum == NPY_UINT64 || typenum == NPY_LONG )
        {
            needcopy = needcast = true;
            new_typenum = NPY_INT;
            type = CV_32S;
        }
        else
        {
            failmsg("%s data type = %d is not supported", info.name, typenum);
            return false;
        }
    }

#ifndef CV_MAX_DIM
    const int CV_MAX_DIM = 32;
#endif

    int ndims = PyArray_NDIM(oarr);
    if(ndims >= CV_MAX_DIM)
    {
        failmsg("%s dimensionality (=%d) is too high", info.name, ndims);
        return false;
    }

    int size[CV_MAX_DIM+1];
    size_t step[CV_MAX_DIM+1];
    size_t elemsize = CV_ELEM_SIZE1(type);
    const npy_intp* _sizes = PyArray_DIMS(oarr);
    const npy_intp* _strides = PyArray_STRIDES(oarr);
    bool ismultichannel = ndims == 3 && _sizes[2] <= CV_CN_MAX;

    for( int i = ndims-1; i >= 0 && !needcopy; i-- )
    {
        // these checks handle cases of
        //  a) multi-dimensional (ndims > 2) arrays, as well as simpler 1- and 2-dimensional cases
        //  b) transposed arrays, where _strides[] elements go in non-descending order
        //  c) flipped arrays, where some of _strides[] elements are negative
        // the _sizes[i] > 1 is needed to avoid spurious copies when NPY_RELAXED_STRIDES is set
        if( (i == ndims-1 && _sizes[i] > 1 && (size_t)_strides[i] != elemsize) ||
            (i < ndims-1 && _sizes[i] > 1 && _strides[i] < _strides[i+1]) )
            needcopy = true;
    }

    if( ismultichannel && _strides[1] != (npy_intp)elemsize*_sizes[2] )
        needcopy = true;

    if (needcopy)
    {
        if (info.outputarg)
        {
            failmsg("Layout of the output array %s is incompatible with cv::Mat (step[ndims-1] != elemsize or step[1] != elemsize*nchannels)", info.name);
            return false;
        }

        if( needcast ) {
            o = PyArray_Cast(oarr, new_typenum);
            oarr = (PyArrayObject*) o;
        }
        else {
            oarr = PyArray_GETCONTIGUOUS(oarr);
            o = (PyObject*) oarr;
        }

        _strides = PyArray_STRIDES(oarr);
    }

    // Normalize strides in case NPY_RELAXED_STRIDES is set
    size_t default_step = elemsize;
    for ( int i = ndims - 1; i >= 0; --i )
    {
        size[i] = (int)_sizes[i];
        if ( size[i] > 1 )
        {
            step[i] = (size_t)_strides[i];
            default_step = step[i] * size[i];
        }
        else
        {
            step[i] = default_step;
            default_step *= size[i];
        }
    }

    // handle degenerate case
    if( ndims == 0) {
        size[ndims] = 1;
        step[ndims] = elemsize;
        ndims++;
    }

    if( ismultichannel )
    {
        ndims--;
        type |= CV_MAKETYPE(0, size[2]);
    }

    if( ndims > 2 && !allowND )
    {
        failmsg("%s has more than 2 dimensions", info.name);
        return false;
    }

    m = Mat(ndims, size, type, PyArray_DATA(oarr), step);
    m.u = g_numpyAllocator.allocate(o, ndims, size, type, step);
    m.addref();

    if( !needcopy )
    {
        Py_INCREF(o);
    }
    m.allocator = &g_numpyAllocator;

    return true;
}

template<typename _Tp, int m, int n>
bool pyopencv_to(PyObject* o, Matx<_Tp, m, n>& mx, const ArgInfo& info)
{
    Mat tmp;
    if (!pyopencv_to(o, tmp, info)) {
        return false;
    }

    tmp.copyTo(mx);
    return true;
}

template<typename _Tp, int cn>
bool pyopencv_to(PyObject* o, Vec<_Tp, cn>& vec, const ArgInfo& info)
{
    return pyopencv_to(o, (Matx<_Tp, cn, 1>&)vec, info);
}

template<>
PyObject* pyopencv_from(const Mat& m)
{
    if( !m.data )
        Py_RETURN_NONE;
    Mat temp, *p = (Mat*)&m;
    if(!p->u || p->allocator != &g_numpyAllocator)
    {
        temp.allocator = &g_numpyAllocator;
        ERRWRAP2(m.copyTo(temp));
        p = &temp;
    }
    PyObject* o = (PyObject*)p->u->userdata;
    Py_INCREF(o);
    return o;
}

template<typename _Tp, int m, int n>
PyObject* pyopencv_from(const Matx<_Tp, m, n>& matx)
{
    return pyopencv_from(Mat(matx));
}

template<typename T>
struct PyOpenCV_Converter< cv::Ptr<T> >
{
    static PyObject* from(const cv::Ptr<T>& p)
    {
        if (!p)
            Py_RETURN_NONE;
        return pyopencv_from(*p);
    }
    static bool to(PyObject *o, Ptr<T>& p, const ArgInfo& info)
    {
        if (!o || o == Py_None)
            return true;
        p = makePtr<T>();
        return pyopencv_to(o, *p, info);
    }
};

template<>
bool pyopencv_to(PyObject* obj, void*& ptr, const ArgInfo& info)
{
    CV_UNUSED(info);
    if (!obj || obj == Py_None)
        return true;

    if (!PyLong_Check(obj))
        return false;
    ptr = PyLong_AsVoidPtr(obj);
    return ptr != NULL && !PyErr_Occurred();
}

static PyObject* pyopencv_from(void*& ptr)
{
    return PyLong_FromVoidPtr(ptr);
}

static bool pyopencv_to(PyObject *o, Scalar& s, const ArgInfo& info)
{
    if(!o || o == Py_None)
        return true;
    if (PySequence_Check(o)) {
        if (4 < PySequence_Size(o))
        {
            failmsg("Scalar value for argument '%s' is longer than 4", info.name);
            return false;
        }
        for (Py_ssize_t i = 0; i < PySequence_Size(o); i++) {
            SafeSeqItem item_wrap(o, i);
            PyObject *item = item_wrap.item;
            if (PyFloat_Check(item) || PyInt_Check(item)) {
                s[(int)i] = PyFloat_AsDouble(item);
            } else {
                failmsg("Scalar value for argument '%s' is not numeric", info.name);
                return false;
            }
        }
    } else {
        if (PyFloat_Check(o) || PyInt_Check(o)) {
            s[0] = PyFloat_AsDouble(o);
        } else {
            failmsg("Scalar value for argument '%s' is not numeric", info.name);
            return false;
        }
    }
    return true;
}

template<>
PyObject* pyopencv_from(const Scalar& src)
{
    return Py_BuildValue("(dddd)", src[0], src[1], src[2], src[3]);
}

template<>
PyObject* pyopencv_from(const bool& value)
{
    return PyBool_FromLong(value);
}

template<>
bool pyopencv_to(PyObject* obj, bool& value, const ArgInfo& info)
{
    if (!obj || obj == Py_None)
    {
        return true;
    }
    if (isBool(obj) || PyArray_IsIntegerScalar(obj))
    {
        npy_bool npy_value = NPY_FALSE;
        const int ret_code = PyArray_BoolConverter(obj, &npy_value);
        if (ret_code >= 0)
        {
            value = (npy_value == NPY_TRUE);
            return true;
        }
    }
    failmsg("Argument '%s' is not convertable to bool", info.name);
    return false;
}

template<>
PyObject* pyopencv_from(const size_t& value)
{
    return PyLong_FromSize_t(value);
}

template<>
bool pyopencv_to(PyObject* obj, size_t& value, const ArgInfo& info)
{
    if (!obj || obj == Py_None)
    {
        return true;
    }
    if (isBool(obj))
    {
        failmsg("Argument '%s' must be integer type, not bool", info.name);
        return false;
    }
    if (PyArray_IsIntegerScalar(obj))
    {
        if (PyLong_Check(obj))
        {
#if defined(CV_PYTHON_3)
            value = PyLong_AsSize_t(obj);
#else
    #if ULONG_MAX == SIZE_MAX
            value = PyLong_AsUnsignedLong(obj);
    #else
            value = PyLong_AsUnsignedLongLong(obj);
    #endif
#endif
        }
#if !defined(CV_PYTHON_3)
        // Python 2.x has PyIntObject which is not a subtype of PyLongObject
        // Overflow check here is unnecessary because object will be converted to long on the
        // interpreter side
        else if (PyInt_Check(obj))
        {
            const long res = PyInt_AsLong(obj);
            if (res < 0) {
                failmsg("Argument '%s' can not be safely parsed to 'size_t'", info.name);
                return false;
            }
    #if ULONG_MAX == SIZE_MAX
            value = PyInt_AsUnsignedLongMask(obj);
    #else
            value = PyInt_AsUnsignedLongLongMask(obj);
    #endif
        }
#endif
        else
        {
            const bool isParsed = parseNumpyScalar<size_t>(obj, value);
            if (!isParsed) {
                failmsg("Argument '%s' can not be safely parsed to 'size_t'", info.name);
                return false;
            }
        }
    }
    else
    {
        failmsg("Argument '%s' is required to be an integer", info.name);
        return false;
    }
    return !PyErr_Occurred();
}

template<>
PyObject* pyopencv_from(const int& value)
{
    return PyInt_FromLong(value);
}

template<>
bool pyopencv_to(PyObject* obj, int& value, const ArgInfo& info)
{
    if (!obj || obj == Py_None)
    {
        return true;
    }
    if (isBool(obj))
    {
        failmsg("Argument '%s' must be integer, not bool", info.name);
        return false;
    }
    if (PyArray_IsIntegerScalar(obj))
    {
        value = PyArray_PyIntAsInt(obj);
    }
    else
    {
        failmsg("Argument '%s' is required to be an integer", info.name);
        return false;
    }
    return !CV_HAS_CONVERSION_ERROR(value);
}

template<>
PyObject* pyopencv_from(const uchar& value)
{
    return PyInt_FromLong(value);
}

template<>
bool pyopencv_to(PyObject* obj, uchar& value, const ArgInfo& info)
{
    CV_UNUSED(info);
    if(!obj || obj == Py_None)
        return true;
    int ivalue = (int)PyInt_AsLong(obj);
    value = cv::saturate_cast<uchar>(ivalue);
    return ivalue != -1 || !PyErr_Occurred();
}

template<>
bool pyopencv_to(PyObject* obj, char& value, const ArgInfo& info)
{
    if (!obj || obj == Py_None)
    {
        return true;
    }
    if (isBool(obj))
    {
        failmsg("Argument '%s' must be an integer, not bool", info.name);
        return false;
    }
    if (PyArray_IsIntegerScalar(obj))
    {
        value = saturate_cast<char>(PyArray_PyIntAsInt(obj));
    }
    else
    {
        failmsg("Argument '%s' is required to be an integer", info.name);
        return false;
    }
    return !CV_HAS_CONVERSION_ERROR(value);
}

template<>
PyObject* pyopencv_from(const double& value)
{
    return PyFloat_FromDouble(value);
}

template<>
bool pyopencv_to(PyObject* obj, double& value, const ArgInfo& info)
{
    if (!obj || obj == Py_None)
    {
        return true;
    }
    if (isBool(obj))
    {
        failmsg("Argument '%s' must be double, not bool", info.name);
        return false;
    }
    if (PyArray_IsPythonNumber(obj))
    {
        if (PyLong_Check(obj))
        {
            value = PyLong_AsDouble(obj);
        }
        else
        {
            value = PyFloat_AsDouble(obj);
        }
    }
    else if (PyArray_CheckScalar(obj))
    {
        const bool isParsed = parseNumpyScalar<double>(obj, value);
        if (!isParsed) {
            failmsg("Argument '%s' can not be safely parsed to 'double'", info.name);
            return false;
        }
    }
    else
    {
        failmsg("Argument '%s' can not be treated as a double", info.name);
        return false;
    }
    return !PyErr_Occurred();
}

template<>
PyObject* pyopencv_from(const float& value)
{
    return PyFloat_FromDouble(value);
}

template<>
bool pyopencv_to(PyObject* obj, float& value, const ArgInfo& info)
{
    if (!obj || obj == Py_None)
    {
        return true;
    }
    if (isBool(obj))
    {
        failmsg("Argument '%s' must be float, not bool", info.name);
        return false;
    }
    if (PyArray_IsPythonNumber(obj))
    {
        if (PyLong_Check(obj))
        {
            double res = PyLong_AsDouble(obj);
            value = static_cast<float>(res);
        }
        else
        {
            double res = PyFloat_AsDouble(obj);
            value = static_cast<float>(res);
        }
    }
    else if (PyArray_CheckScalar(obj))
    {
       const bool isParsed = parseNumpyScalar<float>(obj, value);
        if (!isParsed) {
            failmsg("Argument '%s' can not be safely parsed to 'float'", info.name);
            return false;
        }
    }
    else
    {
        failmsg("Argument '%s' can't be treated as a float", info.name);
        return false;
    }
    return !PyErr_Occurred();
}

template<>
PyObject* pyopencv_from(const int64& value)
{
    return PyLong_FromLongLong(value);
}

template<>
PyObject* pyopencv_from(const String& value)
{
    return PyString_FromString(value.empty() ? "" : value.c_str());
}

#if CV_VERSION_MAJOR == 3
template<>
PyObject* pyopencv_from(const std::string& value)
{
    return PyString_FromString(value.empty() ? "" : value.c_str());
}
#endif

template<>
bool pyopencv_to(PyObject* obj, String &value, const ArgInfo& info)
{
    if(!obj || obj == Py_None)
    {
        return true;
    }
    std::string str;
    if (getUnicodeString(obj, str))
    {
        value = str;
        return true;
    }
    else
    {
        // If error hasn't been already set by Python conversion functions
        if (!PyErr_Occurred())
        {
            // Direct access to underlying slots of PyObjectType is not allowed
            // when limited API is enabled
#ifdef Py_LIMITED_API
            failmsg("Can't convert object to 'str' for '%s'", info.name);
#else
            failmsg("Can't convert object of type '%s' to 'str' for '%s'",
                    obj->ob_type->tp_name, info.name);
#endif
        }
    }
    return false;
}

template<>
bool pyopencv_to(PyObject* obj, Size& sz, const ArgInfo& info)
{
    RefWrapper<int> values[] = {RefWrapper<int>(sz.width),
                                RefWrapper<int>(sz.height)};
    return parseSequence(obj, values, info);
}

template<>
PyObject* pyopencv_from(const Size& sz)
{
    return Py_BuildValue("(ii)", sz.width, sz.height);
}

template<>
bool pyopencv_to(PyObject* obj, Size_<float>& sz, const ArgInfo& info)
{
    RefWrapper<float> values[] = {RefWrapper<float>(sz.width),
                                  RefWrapper<float>(sz.height)};
    return parseSequence(obj, values, info);
}

template<>
PyObject* pyopencv_from(const Size_<float>& sz)
{
    return Py_BuildValue("(ff)", sz.width, sz.height);
}

template<>
bool pyopencv_to(PyObject* obj, Rect& r, const ArgInfo& info)
{
    RefWrapper<int> values[] = {RefWrapper<int>(r.x), RefWrapper<int>(r.y),
                                RefWrapper<int>(r.width),
                                RefWrapper<int>(r.height)};
    return parseSequence(obj, values, info);
}

template<>
PyObject* pyopencv_from(const Rect& r)
{
    return Py_BuildValue("(iiii)", r.x, r.y, r.width, r.height);
}

template<>
bool pyopencv_to(PyObject* obj, Rect2d& r, const ArgInfo& info)
{
    RefWrapper<double> values[] = {
        RefWrapper<double>(r.x), RefWrapper<double>(r.y),
        RefWrapper<double>(r.width), RefWrapper<double>(r.height)};
    return parseSequence(obj, values, info);
}

template<>
PyObject* pyopencv_from(const Rect2d& r)
{
    return Py_BuildValue("(dddd)", r.x, r.y, r.width, r.height);
}

template<>
bool pyopencv_to(PyObject* obj, Range& r, const ArgInfo& info)
{
    if (!obj || obj == Py_None)
    {
        return true;
    }
    if (PyObject_Size(obj) == 0)
    {
        r = Range::all();
        return true;
    }
    RefWrapper<int> values[] = {RefWrapper<int>(r.start), RefWrapper<int>(r.end)};
    return parseSequence(obj, values, info);
}

template<>
PyObject* pyopencv_from(const Range& r)
{
    return Py_BuildValue("(ii)", r.start, r.end);
}

template<>
bool pyopencv_to(PyObject* obj, Point& p, const ArgInfo& info)
{
    RefWrapper<int> values[] = {RefWrapper<int>(p.x), RefWrapper<int>(p.y)};
    return parseSequence(obj, values, info);
}

template <>
bool pyopencv_to(PyObject* obj, Point2f& p, const ArgInfo& info)
{
    RefWrapper<float> values[] = {RefWrapper<float>(p.x),
                                  RefWrapper<float>(p.y)};
    return parseSequence(obj, values, info);
}

template<>
bool pyopencv_to(PyObject* obj, Point2d& p, const ArgInfo& info)
{
    RefWrapper<double> values[] = {RefWrapper<double>(p.x),
                                   RefWrapper<double>(p.y)};
    return parseSequence(obj, values, info);
}

template<>
bool pyopencv_to(PyObject* obj, Point3f& p, const ArgInfo& info)
{
    RefWrapper<float> values[] = {RefWrapper<float>(p.x),
                                  RefWrapper<float>(p.y),
                                  RefWrapper<float>(p.z)};
    return parseSequence(obj, values, info);
}

template<>
bool pyopencv_to(PyObject* obj, Point3d& p, const ArgInfo& info)
{
    RefWrapper<double> values[] = {RefWrapper<double>(p.x),
                                   RefWrapper<double>(p.y),
                                   RefWrapper<double>(p.z)};
    return parseSequence(obj, values, info);
}

template<>
PyObject* pyopencv_from(const Point& p)
{
    return Py_BuildValue("(ii)", p.x, p.y);
}

template<>
PyObject* pyopencv_from(const Point2f& p)
{
    return Py_BuildValue("(dd)", p.x, p.y);
}

template<>
PyObject* pyopencv_from(const Point3f& p)
{
    return Py_BuildValue("(ddd)", p.x, p.y, p.z);
}

static bool pyopencv_to(PyObject* obj, Vec4d& v, ArgInfo& info)
{
    RefWrapper<double> values[] = {RefWrapper<double>(v[0]), RefWrapper<double>(v[1]),
                                   RefWrapper<double>(v[2]), RefWrapper<double>(v[3])};
    return parseSequence(obj, values, info);
}

static bool pyopencv_to(PyObject* obj, Vec4f& v, ArgInfo& info)
{
    RefWrapper<float> values[] = {RefWrapper<float>(v[0]), RefWrapper<float>(v[1]),
                                  RefWrapper<float>(v[2]), RefWrapper<float>(v[3])};
    return parseSequence(obj, values, info);
}

static bool pyopencv_to(PyObject* obj, Vec4i& v, ArgInfo& info)
{
    RefWrapper<int> values[] = {RefWrapper<int>(v[0]), RefWrapper<int>(v[1]),
                                RefWrapper<int>(v[2]), RefWrapper<int>(v[3])};
    return parseSequence(obj, values, info);
}

static bool pyopencv_to(PyObject* obj, Vec3d& v, ArgInfo& info)
{
    RefWrapper<double> values[] = {RefWrapper<double>(v[0]),
                                   RefWrapper<double>(v[1]),
                                   RefWrapper<double>(v[2])};
    return parseSequence(obj, values, info);
}

static bool pyopencv_to(PyObject* obj, Vec3f& v, ArgInfo& info)
{
    RefWrapper<float> values[] = {RefWrapper<float>(v[0]),
                                  RefWrapper<float>(v[1]),
                                  RefWrapper<float>(v[2])};
    return parseSequence(obj, values, info);
}

static bool pyopencv_to(PyObject* obj, Vec3i& v, ArgInfo& info)
{
    RefWrapper<int> values[] = {RefWrapper<int>(v[0]), RefWrapper<int>(v[1]),
                                RefWrapper<int>(v[2])};
    return parseSequence(obj, values, info);
}

static bool pyopencv_to(PyObject* obj, Vec2d& v, ArgInfo& info)
{
    RefWrapper<double> values[] = {RefWrapper<double>(v[0]),
                                   RefWrapper<double>(v[1])};
    return parseSequence(obj, values, info);
}

static bool pyopencv_to(PyObject* obj, Vec2f& v, ArgInfo& info)
{
    RefWrapper<float> values[] = {RefWrapper<float>(v[0]),
                                  RefWrapper<float>(v[1])};
    return parseSequence(obj, values, info);
}

static bool pyopencv_to(PyObject* obj, Vec2i& v, ArgInfo& info)
{
    RefWrapper<int> values[] = {RefWrapper<int>(v[0]), RefWrapper<int>(v[1])};
    return parseSequence(obj, values, info);
}

template<>
PyObject* pyopencv_from(const Vec4d& v)
{
    return Py_BuildValue("(dddd)", v[0], v[1], v[2], v[3]);
}

template<>
PyObject* pyopencv_from(const Vec4f& v)
{
    return Py_BuildValue("(ffff)", v[0], v[1], v[2], v[3]);
}

template<>
PyObject* pyopencv_from(const Vec4i& v)
{
    return Py_BuildValue("(iiii)", v[0], v[1], v[2], v[3]);
}

template<>
PyObject* pyopencv_from(const Vec3d& v)
{
    return Py_BuildValue("(ddd)", v[0], v[1], v[2]);
}

template<>
PyObject* pyopencv_from(const Vec3f& v)
{
    return Py_BuildValue("(fff)", v[0], v[1], v[2]);
}

template<>
PyObject* pyopencv_from(const Vec3i& v)
{
    return Py_BuildValue("(iii)", v[0], v[1], v[2]);
}

template<>
PyObject* pyopencv_from(const Vec2d& v)
{
    return Py_BuildValue("(dd)", v[0], v[1]);
}

template<>
PyObject* pyopencv_from(const Vec2f& v)
{
    return Py_BuildValue("(ff)", v[0], v[1]);
}

template<>
PyObject* pyopencv_from(const Vec2i& v)
{
    return Py_BuildValue("(ii)", v[0], v[1]);
}

template<>
PyObject* pyopencv_from(const Point2d& p)
{
    return Py_BuildValue("(dd)", p.x, p.y);
}

template<>
PyObject* pyopencv_from(const Point3d& p)
{
    return Py_BuildValue("(ddd)", p.x, p.y, p.z);
}

template<>
PyObject* pyopencv_from(const std::pair<int, double>& src)
{
    return Py_BuildValue("(id)", src.first, src.second);
}

template<>
bool pyopencv_to(PyObject* obj, TermCriteria& dst, const ArgInfo& info)
{
    if (!obj || obj == Py_None)
    {
        return true;
    }
    if (!PySequence_Check(obj))
    {
        failmsg("Can't parse '%s' as TermCriteria."
                "Input argument doesn't provide sequence protocol",
                info.name);
        return false;
    }
    const std::size_t sequenceSize = PySequence_Size(obj);
    if (sequenceSize != 3) {
        failmsg("Can't parse '%s' as TermCriteria. Expected sequence length 3, "
                "got %lu",
                info.name, sequenceSize);
        return false;
    }
    {
        const String typeItemName = format("'%s' criteria type", info.name);
        const ArgInfo typeItemInfo(typeItemName.c_str(), false);
        SafeSeqItem typeItem(obj, 0);
        if (!pyopencv_to(typeItem.item, dst.type, typeItemInfo))
        {
            return false;
        }
    }
    {
        const String maxCountItemName = format("'%s' max count", info.name);
        const ArgInfo maxCountItemInfo(maxCountItemName.c_str(), false);
        SafeSeqItem maxCountItem(obj, 1);
        if (!pyopencv_to(maxCountItem.item, dst.maxCount, maxCountItemInfo))
        {
            return false;
        }
    }
    {
        const String epsilonItemName = format("'%s' epsilon", info.name);
        const ArgInfo epsilonItemInfo(epsilonItemName.c_str(), false);
        SafeSeqItem epsilonItem(obj, 2);
        if (!pyopencv_to(epsilonItem.item, dst.epsilon, epsilonItemInfo))
        {
            return false;
        }
    }
    return true;
}

template<>
PyObject* pyopencv_from(const TermCriteria& src)
{
    return Py_BuildValue("(iid)", src.type, src.maxCount, src.epsilon);
}

template<>
bool pyopencv_to(PyObject* obj, RotatedRect& dst, const ArgInfo& info)
{
    if (!obj || obj == Py_None)
    {
        return true;
    }
    if (!PySequence_Check(obj))
    {
        failmsg("Can't parse '%s' as RotatedRect."
                "Input argument doesn't provide sequence protocol",
                info.name);
        return false;
    }
    const std::size_t sequenceSize = PySequence_Size(obj);
    if (sequenceSize != 3)
    {
        failmsg("Can't parse '%s' as RotatedRect. Expected sequence length 3, got %lu",
                info.name, sequenceSize);
        return false;
    }
    {
        const String centerItemName = format("'%s' center point", info.name);
        const ArgInfo centerItemInfo(centerItemName.c_str(), false);
        SafeSeqItem centerItem(obj, 0);
        if (!pyopencv_to(centerItem.item, dst.center, centerItemInfo))
        {
            return false;
        }
    }
    {
        const String sizeItemName = format("'%s' size", info.name);
        const ArgInfo sizeItemInfo(sizeItemName.c_str(), false);
        SafeSeqItem sizeItem(obj, 1);
        if (!pyopencv_to(sizeItem.item, dst.size, sizeItemInfo))
        {
            return false;
        }
    }
    {
        const String angleItemName = format("'%s' angle", info.name);
        const ArgInfo angleItemInfo(angleItemName.c_str(), false);
        SafeSeqItem angleItem(obj, 2);
        if (!pyopencv_to(angleItem.item, dst.angle, angleItemInfo))
        {
            return false;
        }
    }
    return true;
}

template<>
PyObject* pyopencv_from(const RotatedRect& src)
{
    return Py_BuildValue("((ff)(ff)f)", src.center.x, src.center.y, src.size.width, src.size.height, src.angle);
}

template<>
PyObject* pyopencv_from(const Moments& m)
{
    return Py_BuildValue("{s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d}",
                         "m00", m.m00, "m10", m.m10, "m01", m.m01,
                         "m20", m.m20, "m11", m.m11, "m02", m.m02,
                         "m30", m.m30, "m21", m.m21, "m12", m.m12, "m03", m.m03,
                         "mu20", m.mu20, "mu11", m.mu11, "mu02", m.mu02,
                         "mu30", m.mu30, "mu21", m.mu21, "mu12", m.mu12, "mu03", m.mu03,
                         "nu20", m.nu20, "nu11", m.nu11, "nu02", m.nu02,
                         "nu30", m.nu30, "nu21", m.nu21, "nu12", m.nu12, "nu03", m.nu03);
}

template <typename Tp>
struct pyopencvVecConverter;

template <typename Tp>
bool pyopencv_to(PyObject* obj, std::vector<Tp>& value, const ArgInfo& info)
{
    if (!obj || obj == Py_None)
    {
        return true;
    }
    return pyopencvVecConverter<Tp>::to(obj, value, info);
}

template <typename Tp>
PyObject* pyopencv_from(const std::vector<Tp>& value)
{
    return pyopencvVecConverter<Tp>::from(value);
}

template <typename Tp>
static bool pyopencv_to_generic_vec(PyObject* obj, std::vector<Tp>& value, const ArgInfo& info)
{
    if (!obj || obj == Py_None)
    {
        return true;
    }
    if (!PySequence_Check(obj))
    {
        failmsg("Can't parse '%s'. Input argument doesn't provide sequence protocol", info.name);
        return false;
    }
    const size_t n = static_cast<size_t>(PySequence_Size(obj));
    value.resize(n);
    for (size_t i = 0; i < n; i++)
    {
        SafeSeqItem item_wrap(obj, i);
        if (!pyopencv_to(item_wrap.item, value[i], info))
        {
            failmsg("Can't parse '%s'. Sequence item with index %lu has a wrong type", info.name, i);
            return false;
        }
    }
    return true;
}

template <typename Tp>
static PyObject* pyopencv_from_generic_vec(const std::vector<Tp>& value)
{
    Py_ssize_t n = static_cast<Py_ssize_t>(value.size());
    PySafeObject seq(PyTuple_New(n));
    for (Py_ssize_t i = 0; i < n; i++)
    {
        PyObject* item = pyopencv_from(value[i]);
        // If item can't be assigned - PyTuple_SetItem raises exception and returns -1.
        if (!item || PyTuple_SetItem(seq, i, item) == -1)
        {
            return NULL;
        }
    }
    return seq.release();
}

template <typename Tp>
struct pyopencvVecConverter
{
    typedef typename std::vector<Tp>::iterator VecIt;

    static bool to(PyObject* obj, std::vector<Tp>& value, const ArgInfo& info)
    {
        if (!PyArray_Check(obj))
        {
            return pyopencv_to_generic_vec(obj, value, info);
        }
        // If user passed an array it is possible to make faster conversions in several cases
        PyArrayObject* array_obj = reinterpret_cast<PyArrayObject*>(obj);
        const NPY_TYPES target_type = asNumpyType<Tp>();
        const NPY_TYPES source_type = static_cast<NPY_TYPES>(PyArray_TYPE(array_obj));
        if (target_type == NPY_OBJECT)
        {
            // Non-planar arrays representing objects (e.g. array of N Rect is an array of shape Nx4) have NPY_OBJECT
            // as their target type.
            return pyopencv_to_generic_vec(obj, value, info);
        }
        if (PyArray_NDIM(array_obj) > 1)
        {
            failmsg("Can't parse %dD array as '%s' vector argument", PyArray_NDIM(array_obj), info.name);
            return false;
        }
        if (target_type != source_type)
        {
            // Source type requires conversion
            // Allowed conversions for target type is handled in the corresponding pyopencv_to function
            return pyopencv_to_generic_vec(obj, value, info);
        }
        // For all other cases, all array data can be directly copied to std::vector data
        // Simple `memcpy` is not possible because NumPy array can reference a slice of the bigger array:
        // ```
        // arr = np.ones((8, 4, 5), dtype=np.int32)
        // convertible_to_vector_of_int = arr[:, 0, 1]
        // ```
        value.resize(static_cast<size_t>(PyArray_SIZE(array_obj)));
        const npy_intp item_step = PyArray_STRIDE(array_obj, 0) / PyArray_ITEMSIZE(array_obj);
        const Tp* data_ptr = static_cast<Tp*>(PyArray_DATA(array_obj));
        for (VecIt it = value.begin(); it != value.end(); ++it, data_ptr += item_step) {
            *it = *data_ptr;
        }
        return true;
    }

    static PyObject* from(const std::vector<Tp>& value)
    {
        if (value.empty())
        {
            return PyTuple_New(0);
        }
        return from(value, ::traits::IsRepresentableAsMatDataType<Tp>());
    }

private:
    static PyObject* from(const std::vector<Tp>& value, ::traits::FalseType)
    {
        // Underlying type is not representable as Mat Data Type
        return pyopencv_from_generic_vec(value);
    }

    static PyObject* from(const std::vector<Tp>& value, ::traits::TrueType)
    {
        // Underlying type is representable as Mat Data Type, so faster return type is available
        typedef DataType<Tp> DType;
        typedef typename DType::channel_type UnderlyingArrayType;

        // If Mat is always exposed as NumPy array this code path can be reduced to the following snipped:
        //        Mat src(value);
        //        PyObject* array = pyopencv_from(src);
        //        return PyArray_Squeeze(reinterpret_cast<PyArrayObject*>(array));
        // This puts unnecessary restrictions on Mat object those might be avoided without losing the performance.
        // Moreover, this version is a bit faster, because it doesn't create temporary objects with reference counting.

        const NPY_TYPES target_type = asNumpyType<UnderlyingArrayType>();
        const int cols = DType::channels;
        PyObject* array = NULL;
        if (cols == 1)
        {
            npy_intp dims = static_cast<npy_intp>(value.size());
            array = PyArray_SimpleNew(1, &dims, target_type);
        }
        else
        {
            npy_intp dims[2] = {static_cast<npy_intp>(value.size()), cols};
            array = PyArray_SimpleNew(2, dims, target_type);
        }
        if(!array)
        {
            // NumPy arrays with shape (N, 1) and (N) are not equal, so correct error message should distinguish
            // them too.
            String shape;
            if (cols > 1)
            {
                shape = format("(%d x %d)", static_cast<int>(value.size()), cols);
            }
            else
            {
                shape = format("(%d)", static_cast<int>(value.size()));
            }
            const String error_message = format("Can't allocate NumPy array for vector with dtype=%d and shape=%s",
                                                static_cast<int>(target_type), shape.c_str());
            emit_failmsg(PyExc_MemoryError, error_message.c_str());
            return array;
        }
        // Fill the array
        PyArrayObject* array_obj = reinterpret_cast<PyArrayObject*>(array);
        UnderlyingArrayType* array_data = static_cast<UnderlyingArrayType*>(PyArray_DATA(array_obj));
        // if Tp is representable as Mat DataType, so the following cast is pretty safe...
        const UnderlyingArrayType* value_data = reinterpret_cast<const UnderlyingArrayType*>(value.data());
        memcpy(array_data, value_data, sizeof(UnderlyingArrayType) * value.size() * static_cast<size_t>(cols));
        return array;
    }
};

static int OnError(int status, const char *func_name, const char *err_msg, const char *file_name, int line, void *userdata)
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    PyObject *on_error = (PyObject*)userdata;
    PyObject *args = Py_BuildValue("isssi", status, func_name, err_msg, file_name, line);

    PyObject *r = PyObject_Call(on_error, args, NULL);
    if (r == NULL) {
        PyErr_Print();
    } else {
        Py_DECREF(r);
    }

    Py_DECREF(args);
    PyGILState_Release(gstate);

    return 0; // The return value isn't used
}

static PyObject *pycvRedirectError(PyObject*, PyObject *args, PyObject *kw)
{
    const char *keywords[] = { "on_error", NULL };
    PyObject *on_error;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "O", (char**)keywords, &on_error))
        return NULL;

    if ((on_error != Py_None) && !PyCallable_Check(on_error))  {
        PyErr_SetString(PyExc_TypeError, "on_error must be callable");
        return NULL;
    }

    // Keep track of the previous handler parameter, so we can decref it when no longer used
    static PyObject* last_on_error = NULL;
    if (last_on_error) {
        Py_DECREF(last_on_error);
        last_on_error = NULL;
    }

    if (on_error == Py_None) {
        ERRWRAP2(redirectError(NULL));
    } else {
        last_on_error = on_error;
        Py_INCREF(last_on_error);
        ERRWRAP2(redirectError(OnError, last_on_error));
    }
    Py_RETURN_NONE;
}

static void OnMouse(int event, int x, int y, int flags, void* param)
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    PyObject *o = (PyObject*)param;
    PyObject *args = Py_BuildValue("iiiiO", event, x, y, flags, PyTuple_GetItem(o, 1));

    PyObject *r = PyObject_Call(PyTuple_GetItem(o, 0), args, NULL);
    if (r == NULL)
        PyErr_Print();
    else
        Py_DECREF(r);
    Py_DECREF(args);
    PyGILState_Release(gstate);
}

#ifdef HAVE_OPENCV_HIGHGUI
static PyObject *pycvSetMouseCallback(PyObject*, PyObject *args, PyObject *kw)
{
    const char *keywords[] = { "window_name", "on_mouse", "param", NULL };
    char* name;
    PyObject *on_mouse;
    PyObject *param = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "sO|O", (char**)keywords, &name, &on_mouse, &param))
        return NULL;
    if (!PyCallable_Check(on_mouse)) {
        PyErr_SetString(PyExc_TypeError, "on_mouse must be callable");
        return NULL;
    }
    if (param == NULL) {
        param = Py_None;
    }
    PyObject* py_callback_info = Py_BuildValue("OO", on_mouse, param);
    static std::map<std::string, PyObject*> registered_callbacks;
    std::map<std::string, PyObject*>::iterator i = registered_callbacks.find(name);
    if (i != registered_callbacks.end())
    {
        Py_DECREF(i->second);
        i->second = py_callback_info;
    }
    else
    {
        registered_callbacks.insert(std::pair<std::string, PyObject*>(std::string(name), py_callback_info));
    }
    ERRWRAP2(setMouseCallback(name, OnMouse, py_callback_info));
    Py_RETURN_NONE;
}
#endif

static void OnChange(int pos, void *param)
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    PyObject *o = (PyObject*)param;
    PyObject *args = Py_BuildValue("(i)", pos);
    PyObject *r = PyObject_Call(PyTuple_GetItem(o, 0), args, NULL);
    if (r == NULL)
        PyErr_Print();
    else
        Py_DECREF(r);
    Py_DECREF(args);
    PyGILState_Release(gstate);
}

#ifdef HAVE_OPENCV_HIGHGUI
// workaround for #20408, use nullptr, set value later
static int _createTrackbar(const String &trackbar_name, const String &window_name, int value, int count,
                    TrackbarCallback onChange, PyObject* py_callback_info)
{
    int n = createTrackbar(trackbar_name, window_name, NULL, count, onChange, py_callback_info);
    setTrackbarPos(trackbar_name, window_name, value);
    return n;
}
static PyObject *pycvCreateTrackbar(PyObject*, PyObject *args)
{
    PyObject *on_change;
    char* trackbar_name;
    char* window_name;
    int value;
    int count;

    if (!PyArg_ParseTuple(args, "ssiiO", &trackbar_name, &window_name, &value, &count, &on_change))
        return NULL;
    if (!PyCallable_Check(on_change)) {
        PyErr_SetString(PyExc_TypeError, "on_change must be callable");
        return NULL;
    }
    PyObject* py_callback_info = Py_BuildValue("OO", on_change, Py_None);
    std::string name = std::string(window_name) + ":" + std::string(trackbar_name);
    static std::map<std::string, PyObject*> registered_callbacks;
    std::map<std::string, PyObject*>::iterator i = registered_callbacks.find(name);
    if (i != registered_callbacks.end())
    {
        Py_DECREF(i->second);
        i->second = py_callback_info;
    }
    else
    {
        registered_callbacks.insert(std::pair<std::string, PyObject*>(name, py_callback_info));
    }
    ERRWRAP2(_createTrackbar(trackbar_name, window_name, value, count, OnChange, py_callback_info));
    Py_RETURN_NONE;
}

static void OnButtonChange(int state, void *param)
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    PyObject *o = (PyObject*)param;
    PyObject *args;
    if(PyTuple_GetItem(o, 1) != NULL)
    {
        args = Py_BuildValue("(iO)", state, PyTuple_GetItem(o,1));
    }
    else
    {
        args = Py_BuildValue("(i)", state);
    }

    PyObject *r = PyObject_Call(PyTuple_GetItem(o, 0), args, NULL);
    if (r == NULL)
        PyErr_Print();
    else
        Py_DECREF(r);
    Py_DECREF(args);
    PyGILState_Release(gstate);
}

static PyObject *pycvCreateButton(PyObject*, PyObject *args, PyObject *kw)
{
    const char* keywords[] = {"buttonName", "onChange", "userData", "buttonType", "initialButtonState", NULL};
    PyObject *on_change;
    PyObject *userdata = NULL;
    char* button_name;
    int button_type = 0;
    int initial_button_state = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "sO|Oii", (char**)keywords, &button_name, &on_change, &userdata, &button_type, &initial_button_state))
        return NULL;
    if (!PyCallable_Check(on_change)) {
        PyErr_SetString(PyExc_TypeError, "onChange must be callable");
        return NULL;
    }
    if (userdata == NULL) {
        userdata = Py_None;
    }

    PyObject* py_callback_info = Py_BuildValue("OO", on_change, userdata);
    std::string name(button_name);

    static std::map<std::string, PyObject*> registered_callbacks;
    std::map<std::string, PyObject*>::iterator i = registered_callbacks.find(name);
    if (i != registered_callbacks.end())
    {
        Py_DECREF(i->second);
        i->second = py_callback_info;
    }
    else
    {
        registered_callbacks.insert(std::pair<std::string, PyObject*>(name, py_callback_info));
    }
    ERRWRAP2(createButton(button_name, OnButtonChange, py_callback_info, button_type, initial_button_state != 0));
    Py_RETURN_NONE;
}
#endif

///////////////////////////////////////////////////////////////////////////////////////

static int convert_to_char(PyObject *o, char *dst, const ArgInfo& info)
{
    std::string str;
    if (getUnicodeString(o, str))
    {
        *dst = str[0];
        return 1;
    }
    (*dst) = 0;
    return failmsg("Expected single character string for argument '%s'", info.name);
}

#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wunused-parameter"
#  pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif


#include "pyopencv_generated_enums.h"
#include "pyopencv_custom_headers.h"

#ifdef CVPY_DYNAMIC_INIT
#define CVPY_TYPE(WNAME, NAME, STORAGE, SNAME, _1, _2) CVPY_TYPE_DECLARE_DYNAMIC(WNAME, NAME, STORAGE, SNAME)
#else
#define CVPY_TYPE(WNAME, NAME, STORAGE, SNAME, _1, _2) CVPY_TYPE_DECLARE(WNAME, NAME, STORAGE, SNAME)
#endif
#include "pyopencv_generated_types.h"
#undef CVPY_TYPE

#include "pyopencv_generated_types_content.h"
#include "pyopencv_generated_funcs.h"


static PyMethodDef special_methods[] = {
  {"redirectError", CV_PY_FN_WITH_KW(pycvRedirectError), "redirectError(onError) -> None"},
#ifdef HAVE_OPENCV_HIGHGUI
  {"createTrackbar", (PyCFunction)pycvCreateTrackbar, METH_VARARGS, "createTrackbar(trackbarName, windowName, value, count, onChange) -> None"},
  {"createButton", CV_PY_FN_WITH_KW(pycvCreateButton), "createButton(buttonName, onChange [, userData, buttonType, initialButtonState]) -> None"},
  {"setMouseCallback", CV_PY_FN_WITH_KW(pycvSetMouseCallback), "setMouseCallback(windowName, onMouse [, param]) -> None"},
#endif
#ifdef HAVE_OPENCV_DNN
  {"dnn_registerLayer", CV_PY_FN_WITH_KW(pyopencv_cv_dnn_registerLayer), "registerLayer(type, class) -> None"},
  {"dnn_unregisterLayer", CV_PY_FN_WITH_KW(pyopencv_cv_dnn_unregisterLayer), "unregisterLayer(type) -> None"},
#endif
  {NULL, NULL},
};

/************************************************************************/
/* Module init */

struct ConstDef
{
    const char * name;
    long long val;
};

static void init_submodule(PyObject * root, const char * name, PyMethodDef * methods, ConstDef * consts)
{
  // traverse and create nested submodules
  std::string s = name;
  size_t i = s.find('.');
  while (i < s.length() && i != std::string::npos)
  {
    size_t j = s.find('.', i);
    if (j == std::string::npos)
        j = s.length();
    std::string short_name = s.substr(i, j-i);
    std::string full_name = s.substr(0, j);
    i = j+1;

    PyObject * d = PyModule_GetDict(root);
    PyObject * submod = PyDict_GetItemString(d, short_name.c_str());
    if (submod == NULL)
    {
        submod = PyImport_AddModule(full_name.c_str());
        PyDict_SetItemString(d, short_name.c_str(), submod);
    }

    if (short_name != "")
        root = submod;
  }

  // populate module's dict
  PyObject * d = PyModule_GetDict(root);
  for (PyMethodDef * m = methods; m->ml_name != NULL; ++m)
  {
    PyObject * method_obj = PyCFunction_NewEx(m, NULL, NULL);
    PyDict_SetItemString(d, m->ml_name, method_obj);
    Py_DECREF(method_obj);
  }
  for (ConstDef * c = consts; c->name != NULL; ++c)
  {
    PyDict_SetItemString(d, c->name, PyLong_FromLongLong(c->val));
  }

}

#include "pyopencv_generated_modules_content.h"

static bool init_body(PyObject * m)
{
#define CVPY_MODULE(NAMESTR, NAME) \
    init_submodule(m, MODULESTR NAMESTR, methods_##NAME, consts_##NAME)
    #include "pyopencv_generated_modules.h"
#undef CVPY_MODULE

#ifdef CVPY_DYNAMIC_INIT
#define CVPY_TYPE(WNAME, NAME, _1, _2, BASE, CONSTRUCTOR) CVPY_TYPE_INIT_DYNAMIC(WNAME, NAME, return false, BASE, CONSTRUCTOR)
    PyObject * pyopencv_NoBase_TypePtr = NULL;
#else
#define CVPY_TYPE(WNAME, NAME, _1, _2, BASE, CONSTRUCTOR) CVPY_TYPE_INIT_STATIC(WNAME, NAME, return false, BASE, CONSTRUCTOR)
    PyTypeObject * pyopencv_NoBase_TypePtr = NULL;
#endif
    #include "pyopencv_generated_types.h"
#undef CVPY_TYPE

    PyObject* d = PyModule_GetDict(m);


    PyDict_SetItemString(d, "__version__", PyString_FromString(CV_VERSION));

    PyObject *opencv_error_dict = PyDict_New();
    PyDict_SetItemString(opencv_error_dict, "file", Py_None);
    PyDict_SetItemString(opencv_error_dict, "func", Py_None);
    PyDict_SetItemString(opencv_error_dict, "line", Py_None);
    PyDict_SetItemString(opencv_error_dict, "code", Py_None);
    PyDict_SetItemString(opencv_error_dict, "msg", Py_None);
    PyDict_SetItemString(opencv_error_dict, "err", Py_None);
    opencv_error = PyErr_NewException((char*)MODULESTR".error", NULL, opencv_error_dict);
    Py_DECREF(opencv_error_dict);
    PyDict_SetItemString(d, "error", opencv_error);


#define PUBLISH(I) PyDict_SetItemString(d, #I, PyInt_FromLong(I))
    PUBLISH(CV_8U);
    PUBLISH(CV_8UC1);
    PUBLISH(CV_8UC2);
    PUBLISH(CV_8UC3);
    PUBLISH(CV_8UC4);
    PUBLISH(CV_8S);
    PUBLISH(CV_8SC1);
    PUBLISH(CV_8SC2);
    PUBLISH(CV_8SC3);
    PUBLISH(CV_8SC4);
    PUBLISH(CV_16U);
    PUBLISH(CV_16UC1);
    PUBLISH(CV_16UC2);
    PUBLISH(CV_16UC3);
    PUBLISH(CV_16UC4);
    PUBLISH(CV_16S);
    PUBLISH(CV_16SC1);
    PUBLISH(CV_16SC2);
    PUBLISH(CV_16SC3);
    PUBLISH(CV_16SC4);
    PUBLISH(CV_32S);
    PUBLISH(CV_32SC1);
    PUBLISH(CV_32SC2);
    PUBLISH(CV_32SC3);
    PUBLISH(CV_32SC4);
    PUBLISH(CV_32F);
    PUBLISH(CV_32FC1);
    PUBLISH(CV_32FC2);
    PUBLISH(CV_32FC3);
    PUBLISH(CV_32FC4);
    PUBLISH(CV_64F);
    PUBLISH(CV_64FC1);
    PUBLISH(CV_64FC2);
    PUBLISH(CV_64FC3);
    PUBLISH(CV_64FC4);
#undef PUBLISH

    return true;
}

#if defined(__GNUC__)
#pragma GCC visibility push(default)
#endif

#if defined(CV_PYTHON_3)
// === Python 3

static struct PyModuleDef cv2_moduledef =
{
    PyModuleDef_HEAD_INIT,
    MODULESTR,
    "Python wrapper for OpenCV.",
    -1,     /* size of per-interpreter state of the module,
               or -1 if the module keeps state in global variables. */
    special_methods
};

PyMODINIT_FUNC PyInit_cv2();
PyObject* PyInit_cv2()
{
    import_array(); // from numpy
    PyObject* m = PyModule_Create(&cv2_moduledef);
    if (!init_body(m))
        return NULL;
    return m;
}

#else
// === Python 2
PyMODINIT_FUNC initcv2();
void initcv2()
{
    import_array(); // from numpy
    PyObject* m = Py_InitModule(MODULESTR, special_methods);
    init_body(m);
}

#endif
