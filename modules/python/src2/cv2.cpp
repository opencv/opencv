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

#include <math.h>
#include <Python.h>

#if PY_MAJOR_VERSION < 3
#undef CVPY_DYNAMIC_INIT
#endif

#if defined(_MSC_VER) && (_MSC_VER > 1800)
#pragma warning(pop)
#endif

#define MODULESTR "cv2"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/ndarrayobject.h>

#include "pyopencv_generated_include.h"
#include "opencv2/core/types_c.h"
#include "opencv2/opencv_modules.hpp"
#include "pycompat.hpp"
#include <map>

#include <type_traits>  // std::enable_if

template<typename T, class TEnable = void>  // TEnable is used for SFINAE checks
struct PyOpenCV_Converter
{
    //static inline bool to(PyObject* obj, T& p, const char* name);
    //static inline PyObject* from(const T& src);
};

template<typename T> static
bool pyopencv_to(PyObject* obj, T& p, const char* name = "<unknown>") { return PyOpenCV_Converter<T>::to(obj, p, name); }

template<typename T> static
PyObject* pyopencv_from(const T& src) { return PyOpenCV_Converter<T>::from(src); }

static PyObject* opencv_error = NULL;

static int failmsg(const char *fmt, ...)
{
    char str[1000];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(str, sizeof(str), fmt, ap);
    va_end(ap);

    PyErr_SetString(PyExc_TypeError, str);
    return 0;
}

struct ArgInfo
{
    const char * name;
    bool outputarg;
    // more fields may be added if necessary

    ArgInfo(const char * name_, bool outputarg_)
        : name(name_)
        , outputarg(outputarg_) {}

    // to match with older pyopencv_to function signature
    operator const char *() const { return name; }
};

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

#define ERRWRAP2(expr) \
try \
{ \
    PyAllowThreads allowThreads; \
    expr; \
} \
catch (const cv::Exception &e) \
{ \
    PyObject_SetAttrString(opencv_error, "file", PyString_FromString(e.file.c_str())); \
    PyObject_SetAttrString(opencv_error, "func", PyString_FromString(e.func.c_str())); \
    PyObject_SetAttrString(opencv_error, "line", PyInt_FromLong(e.line)); \
    PyObject_SetAttrString(opencv_error, "code", PyInt_FromLong(e.code)); \
    PyObject_SetAttrString(opencv_error, "msg", PyString_FromString(e.msg.c_str())); \
    PyObject_SetAttrString(opencv_error, "err", PyString_FromString(e.err.c_str())); \
    PyErr_SetString(opencv_error, e.what()); \
    return 0; \
}

using namespace cv;

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

static PyObject* failmsgp(const char *fmt, ...)
{
  char str[1000];

  va_list ap;
  va_start(ap, fmt);
  vsnprintf(str, sizeof(str), fmt, ap);
  va_end(ap);

  PyErr_SetString(PyExc_TypeError, str);
  return 0;
}

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

    UMatData* allocate(int dims0, const int* sizes, int type, void* data, size_t* step, AccessFlag flags, UMatUsageFlags usageFlags) const CV_OVERRIDE
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

    bool allocate(UMatData* u, AccessFlag accessFlags, UMatUsageFlags usageFlags) const CV_OVERRIDE
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

// special case, when the converter needs full ArgInfo structure
static bool pyopencv_to(PyObject* o, Mat& m, const ArgInfo info)
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

template<>
bool pyopencv_to(PyObject* o, Mat& m, const char* name)
{
    return pyopencv_to(o, m, ArgInfo(name, 0));
}

template<typename _Tp, int m, int n>
bool pyopencv_to(PyObject* o, Matx<_Tp, m, n>& mx, const ArgInfo info)
{
    Mat tmp;
    if (!pyopencv_to(o, tmp, info)) {
        return false;
    }

    tmp.copyTo(mx);
    return true;
}

template<typename _Tp, int m, int n>
bool pyopencv_to(PyObject* o, Matx<_Tp, m, n>& mx, const char* name)
{
    return pyopencv_to(o, mx, ArgInfo(name, 0));
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
    static bool to(PyObject *o, Ptr<T>& p, const char *name)
    {
        if (!o || o == Py_None)
            return true;
        p = makePtr<T>();
        return pyopencv_to(o, *p, name);
    }
};

template<>
bool pyopencv_to(PyObject* obj, void*& ptr, const char* name)
{
    CV_UNUSED(name);
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

struct SafeSeqItem
{
    PyObject * item;
    SafeSeqItem(PyObject *obj, size_t idx) { item = PySequence_GetItem(obj, idx); }
    ~SafeSeqItem() { Py_XDECREF(item); }
};

static bool pyopencv_to(PyObject *o, Scalar& s, const ArgInfo info)
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
bool pyopencv_to(PyObject *o, Scalar& s, const char *name)
{
    return pyopencv_to(o, s, ArgInfo(name, 0));
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
bool pyopencv_to(PyObject* obj, bool& value, const char* name)
{
    CV_UNUSED(name);
    if(!obj || obj == Py_None)
        return true;
    int _val = PyObject_IsTrue(obj);
    if(_val < 0)
        return false;
    value = _val > 0;
    return true;
}

template<>
PyObject* pyopencv_from(const size_t& value)
{
    return PyLong_FromSize_t(value);
}

template<>
bool pyopencv_to(PyObject* obj, size_t& value, const char* name)
{
    CV_UNUSED(name);
    if(!obj || obj == Py_None)
        return true;
    value = (int)PyLong_AsUnsignedLong(obj);
    return value != (size_t)-1 || !PyErr_Occurred();
}

template<>
PyObject* pyopencv_from(const int& value)
{
    return PyInt_FromLong(value);
}

template<>
bool pyopencv_to(PyObject* obj, int& value, const char* name)
{
    CV_UNUSED(name);
    if(!obj || obj == Py_None)
        return true;
    if(PyInt_Check(obj))
        value = (int)PyInt_AsLong(obj);
    else if(PyLong_Check(obj))
        value = (int)PyLong_AsLong(obj);
    else
        return false;
    return value != -1 || !PyErr_Occurred();
}

// There is conflict between "size_t" and "unsigned int".
// They are the same type on some 32-bit platforms.
template<typename T>
struct PyOpenCV_Converter
    < T, typename std::enable_if< std::is_same<unsigned int, T>::value && !std::is_same<unsigned int, size_t>::value >::type >
{
    static inline PyObject* from(const unsigned int& value)
    {
        return PyLong_FromUnsignedLong(value);
    }

    static inline bool to(PyObject* obj, unsigned int& value, const char* name)
    {
        CV_UNUSED(name);
        if(!obj || obj == Py_None)
            return true;
        if(PyInt_Check(obj))
            value = (unsigned int)PyInt_AsLong(obj);
        else if(PyLong_Check(obj))
            value = (unsigned int)PyLong_AsLong(obj);
        else
            return false;
        return value != (unsigned int)-1 || !PyErr_Occurred();
    }
};

template<>
PyObject* pyopencv_from(const uchar& value)
{
    return PyInt_FromLong(value);
}

template<>
bool pyopencv_to(PyObject* obj, uchar& value, const char* name)
{
    CV_UNUSED(name);
    if(!obj || obj == Py_None)
        return true;
    int ivalue = (int)PyInt_AsLong(obj);
    value = cv::saturate_cast<uchar>(ivalue);
    return ivalue != -1 || !PyErr_Occurred();
}

template<>
PyObject* pyopencv_from(const double& value)
{
    return PyFloat_FromDouble(value);
}

template<>
bool pyopencv_to(PyObject* obj, double& value, const char* name)
{
    CV_UNUSED(name);
    if(!obj || obj == Py_None)
        return true;
    if(!!PyInt_CheckExact(obj))
        value = (double)PyInt_AS_LONG(obj);
    else
        value = PyFloat_AsDouble(obj);
    return !PyErr_Occurred();
}

template<>
PyObject* pyopencv_from(const float& value)
{
    return PyFloat_FromDouble(value);
}

template<>
bool pyopencv_to(PyObject* obj, float& value, const char* name)
{
    CV_UNUSED(name);
    if(!obj || obj == Py_None)
        return true;
    if(!!PyInt_CheckExact(obj))
        value = (float)PyInt_AS_LONG(obj);
    else
        value = (float)PyFloat_AsDouble(obj);
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

template<>
bool pyopencv_to(PyObject* obj, String &value, const char* name)
{
    CV_UNUSED(name);
    if(!obj || obj == Py_None)
        return true;
    std::string str;
    if (getUnicodeString(obj, str))
    {
        value = str;
        return true;
    }
    return false;
}

template<>
bool pyopencv_to(PyObject* obj, Size& sz, const char* name)
{
    CV_UNUSED(name);
    if(!obj || obj == Py_None)
        return true;
    return PyArg_ParseTuple(obj, "ii", &sz.width, &sz.height) > 0;
}

template<>
PyObject* pyopencv_from(const Size& sz)
{
    return Py_BuildValue("(ii)", sz.width, sz.height);
}

template<>
bool pyopencv_to(PyObject* obj, Size_<float>& sz, const char* name)
{
    CV_UNUSED(name);
    if(!obj || obj == Py_None)
        return true;
    return PyArg_ParseTuple(obj, "ff", &sz.width, &sz.height) > 0;
}

template<>
PyObject* pyopencv_from(const Size_<float>& sz)
{
    return Py_BuildValue("(ff)", sz.width, sz.height);
}

template<>
bool pyopencv_to(PyObject* obj, Rect& r, const char* name)
{
    CV_UNUSED(name);
    if(!obj || obj == Py_None)
        return true;
    return PyArg_ParseTuple(obj, "iiii", &r.x, &r.y, &r.width, &r.height) > 0;
}

template<>
PyObject* pyopencv_from(const Rect& r)
{
    return Py_BuildValue("(iiii)", r.x, r.y, r.width, r.height);
}

template<>
bool pyopencv_to(PyObject* obj, Rect2d& r, const char* name)
{
    CV_UNUSED(name);
    if(!obj || obj == Py_None)
        return true;
    return PyArg_ParseTuple(obj, "dddd", &r.x, &r.y, &r.width, &r.height) > 0;
}

template<>
PyObject* pyopencv_from(const Rect2d& r)
{
    return Py_BuildValue("(dddd)", r.x, r.y, r.width, r.height);
}

template<>
bool pyopencv_to(PyObject* obj, Range& r, const char* name)
{
    CV_UNUSED(name);
    if(!obj || obj == Py_None)
        return true;
    while (PySequence_Check(obj))
    {
        if (2 != PySequence_Size(obj))
        {
            failmsg("Range value for argument '%s' is longer than 2", name);
            return false;
        }
        {
            SafeSeqItem item_wrap(obj, 0);
            PyObject *item = item_wrap.item;
            if (PyInt_Check(item)) {
                r.start = (int)PyInt_AsLong(item);
            } else {
                failmsg("Range.start value for argument '%s' is not integer", name);
                break;
            }
        }
        {
            SafeSeqItem item_wrap(obj, 1);
            PyObject *item = item_wrap.item;
            if (PyInt_Check(item)) {
                r.end = (int)PyInt_AsLong(item);
            } else {
                failmsg("Range.end value for argument '%s' is not integer", name);
                break;
            }
        }
        return true;
    }
    if(PyObject_Size(obj) == 0)
    {
        r = Range::all();
        return true;
    }
    return PyArg_ParseTuple(obj, "ii", &r.start, &r.end) > 0;
}

template<>
PyObject* pyopencv_from(const Range& r)
{
    return Py_BuildValue("(ii)", r.start, r.end);
}

template<>
bool pyopencv_to(PyObject* obj, Point& p, const char* name)
{
    CV_UNUSED(name);
    if(!obj || obj == Py_None)
        return true;
    if(PyComplex_Check(obj))
    {
        p.x = saturate_cast<int>(PyComplex_RealAsDouble(obj));
        p.y = saturate_cast<int>(PyComplex_ImagAsDouble(obj));
        return true;
    }
    return PyArg_ParseTuple(obj, "ii", &p.x, &p.y) > 0;
}

template<>
bool pyopencv_to(PyObject* obj, Point2f& p, const char* name)
{
    CV_UNUSED(name);
    if(!obj || obj == Py_None)
        return true;
    if (PyComplex_Check(obj))
    {
        p.x = saturate_cast<float>(PyComplex_RealAsDouble(obj));
        p.y = saturate_cast<float>(PyComplex_ImagAsDouble(obj));
        return true;
    }
    return PyArg_ParseTuple(obj, "ff", &p.x, &p.y) > 0;
}

template<>
bool pyopencv_to(PyObject* obj, Point2d& p, const char* name)
{
    CV_UNUSED(name);
    if(!obj || obj == Py_None)
        return true;
    if(PyComplex_Check(obj))
    {
        p.x = PyComplex_RealAsDouble(obj);
        p.y = PyComplex_ImagAsDouble(obj);
        return true;
    }
    return PyArg_ParseTuple(obj, "dd", &p.x, &p.y) > 0;
}

template<>
bool pyopencv_to(PyObject* obj, Point3f& p, const char* name)
{
    CV_UNUSED(name);
    if(!obj || obj == Py_None)
        return true;
    return PyArg_ParseTuple(obj, "fff", &p.x, &p.y, &p.z) > 0;
}

template<>
bool pyopencv_to(PyObject* obj, Point3d& p, const char* name)
{
    CV_UNUSED(name);
    if(!obj || obj == Py_None)
        return true;
    return PyArg_ParseTuple(obj, "ddd", &p.x, &p.y, &p.z) > 0;
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

static bool pyopencv_to(PyObject* obj, Vec4d& v, ArgInfo info)
{
    CV_UNUSED(info);
    if (!obj)
        return true;
    return PyArg_ParseTuple(obj, "dddd", &v[0], &v[1], &v[2], &v[3]) > 0;
}
template<>
bool pyopencv_to(PyObject* obj, Vec4d& v, const char* name)
{
    return pyopencv_to(obj, v, ArgInfo(name, 0));
}

static bool pyopencv_to(PyObject* obj, Vec4f& v, ArgInfo info)
{
    CV_UNUSED(info);
    if (!obj)
        return true;
    return PyArg_ParseTuple(obj, "ffff", &v[0], &v[1], &v[2], &v[3]) > 0;
}
template<>
bool pyopencv_to(PyObject* obj, Vec4f& v, const char* name)
{
    return pyopencv_to(obj, v, ArgInfo(name, 0));
}

static bool pyopencv_to(PyObject* obj, Vec4i& v, ArgInfo info)
{
    CV_UNUSED(info);
    if (!obj)
        return true;
    return PyArg_ParseTuple(obj, "iiii", &v[0], &v[1], &v[2], &v[3]) > 0;
}
template<>
bool pyopencv_to(PyObject* obj, Vec4i& v, const char* name)
{
    return pyopencv_to(obj, v, ArgInfo(name, 0));
}

static bool pyopencv_to(PyObject* obj, Vec3d& v, ArgInfo info)
{
    CV_UNUSED(info);
    if (!obj)
        return true;
    return PyArg_ParseTuple(obj, "ddd", &v[0], &v[1], &v[2]) > 0;
}
template<>
bool pyopencv_to(PyObject* obj, Vec3d& v, const char* name)
{
    return pyopencv_to(obj, v, ArgInfo(name, 0));
}

static bool pyopencv_to(PyObject* obj, Vec3f& v, ArgInfo info)
{
    CV_UNUSED(info);
    if (!obj)
        return true;
    return PyArg_ParseTuple(obj, "fff", &v[0], &v[1], &v[2]) > 0;
}
template<>
bool pyopencv_to(PyObject* obj, Vec3f& v, const char* name)
{
    return pyopencv_to(obj, v, ArgInfo(name, 0));
}

static bool pyopencv_to(PyObject* obj, Vec3i& v, ArgInfo info)
{
    CV_UNUSED(info);
    if (!obj)
        return true;
    return PyArg_ParseTuple(obj, "iii", &v[0], &v[1], &v[2]) > 0;
}
template<>
bool pyopencv_to(PyObject* obj, Vec3i& v, const char* name)
{
    return pyopencv_to(obj, v, ArgInfo(name, 0));
}

static bool pyopencv_to(PyObject* obj, Vec2d& v, ArgInfo info)
{
    CV_UNUSED(info);
    if (!obj)
        return true;
    return PyArg_ParseTuple(obj, "dd", &v[0], &v[1]) > 0;
}
template<>
bool pyopencv_to(PyObject* obj, Vec2d& v, const char* name)
{
    return pyopencv_to(obj, v, ArgInfo(name, 0));
}

static bool pyopencv_to(PyObject* obj, Vec2f& v, ArgInfo info)
{
    CV_UNUSED(info);
    if (!obj)
        return true;
    return PyArg_ParseTuple(obj, "ff", &v[0], &v[1]) > 0;
}
template<>
bool pyopencv_to(PyObject* obj, Vec2f& v, const char* name)
{
    return pyopencv_to(obj, v, ArgInfo(name, 0));
}

static bool pyopencv_to(PyObject* obj, Vec2i& v, ArgInfo info)
{
    CV_UNUSED(info);
    if (!obj)
        return true;
    return PyArg_ParseTuple(obj, "ii", &v[0], &v[1]) > 0;
}
template<>
bool pyopencv_to(PyObject* obj, Vec2i& v, const char* name)
{
    return pyopencv_to(obj, v, ArgInfo(name, 0));
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

template<typename _Tp> struct pyopencvVecConverter
{
    typedef typename DataType<_Tp>::channel_type _Cp;
    static inline bool copyOneItem(PyObject *obj, size_t start, int channels, _Cp * data)
    {
        for(size_t j = 0; (int)j < channels; j++ )
        {
            SafeSeqItem sub_item_wrap(obj, start + j);
            PyObject* item_ij = sub_item_wrap.item;
            if( PyInt_Check(item_ij))
            {
                int v = (int)PyInt_AsLong(item_ij);
                if( v == -1 && PyErr_Occurred() )
                    return false;
                data[j] = saturate_cast<_Cp>(v);
            }
            else if( PyLong_Check(item_ij))
            {
                int v = (int)PyLong_AsLong(item_ij);
                if( v == -1 && PyErr_Occurred() )
                    return false;
                data[j] = saturate_cast<_Cp>(v);
            }
            else if( PyFloat_Check(item_ij))
            {
                double v = PyFloat_AsDouble(item_ij);
                if( PyErr_Occurred() )
                    return false;
                data[j] = saturate_cast<_Cp>(v);
            }
            else
                return false;
        }
        return true;
    }
    static bool to(PyObject* obj, std::vector<_Tp>& value, const ArgInfo info)
    {
        if(!obj || obj == Py_None)
            return true;
        if (PyArray_Check(obj))
        {
            Mat m;
            pyopencv_to(obj, m, info);
            m.copyTo(value);
            return true;
        }
        else if (PySequence_Check(obj))
        {
            const int type = traits::Type<_Tp>::value;
            const int depth = CV_MAT_DEPTH(type), channels = CV_MAT_CN(type);
            size_t i, n = PySequence_Size(obj);
            value.resize(n);
            for (i = 0; i < n; i++ )
            {
                SafeSeqItem item_wrap(obj, i);
                PyObject* item = item_wrap.item;
                _Cp* data = (_Cp*)&value[i];

                if( channels == 2 && PyComplex_Check(item) )
                {
                    data[0] = saturate_cast<_Cp>(PyComplex_RealAsDouble(item));
                    data[1] = saturate_cast<_Cp>(PyComplex_ImagAsDouble(item));
                }
                else if( channels > 1 )
                {
                    if( PyArray_Check(item))
                    {
                        Mat src;
                        pyopencv_to(item, src, info);
                        if( src.dims != 2 || src.channels() != 1 ||
                           ((src.cols != 1 || src.rows != channels) &&
                            (src.cols != channels || src.rows != 1)))
                            break;
                        Mat dst(src.rows, src.cols, depth, data);
                        src.convertTo(dst, type);
                        if( dst.data != (uchar*)data )
                            break;
                    }
                    else if (PySequence_Check(item))
                    {
                        if (!copyOneItem(item, 0, channels, data))
                            break;
                    }
                    else
                    {
                        break;
                    }
                }
                else if (channels == 1)
                {
                    if (!copyOneItem(obj, i, channels, data))
                        break;
                }
                else
                {
                    break;
                }
            }
            return i == n;
        }
        return false;
    }

    static PyObject* from(const std::vector<_Tp>& value)
    {
        if(value.empty())
            return PyTuple_New(0);
        int type = traits::Type<_Tp>::value;
        int depth = CV_MAT_DEPTH(type), channels = CV_MAT_CN(type);
        Mat src((int)value.size(), channels, depth, (uchar*)&value[0]);
        return pyopencv_from(src);
    }
};

template<typename _Tp>
bool pyopencv_to(PyObject* obj, std::vector<_Tp>& value, const ArgInfo info)
{
    return pyopencvVecConverter<_Tp>::to(obj, value, info);
}

template<typename _Tp>
PyObject* pyopencv_from(const std::vector<_Tp>& value)
{
    return pyopencvVecConverter<_Tp>::from(value);
}

template<typename _Tp> static inline bool pyopencv_to_generic_vec(PyObject* obj, std::vector<_Tp>& value, const ArgInfo info)
{
    if(!obj || obj == Py_None)
       return true;
    if (!PySequence_Check(obj))
        return false;
    size_t n = PySequence_Size(obj);
    value.resize(n);
    for(size_t i = 0; i < n; i++ )
    {
        SafeSeqItem item_wrap(obj, i);
        if(!pyopencv_to(item_wrap.item, value[i], info))
            return false;
    }
    return true;
}

template<typename _Tp> static inline PyObject* pyopencv_from_generic_vec(const std::vector<_Tp>& value)
{
    int i, n = (int)value.size();
    PyObject* seq = PyList_New(n);
    for( i = 0; i < n; i++ )
    {
        PyObject* item = pyopencv_from(value[i]);
        if(!item)
            break;
        PyList_SetItem(seq, i, item);
    }
    if( i < n )
    {
        Py_DECREF(seq);
        return 0;
    }
    return seq;
}

template<>
PyObject* pyopencv_from(const std::pair<int, double>& src)
{
    return Py_BuildValue("(id)", src.first, src.second);
}

template<typename _Tp, typename _Tr> struct pyopencvVecConverter<std::pair<_Tp, _Tr> >
{
    static bool to(PyObject* obj, std::vector<std::pair<_Tp, _Tr> >& value, const ArgInfo info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const std::vector<std::pair<_Tp, _Tr> >& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

template<typename _Tp> struct pyopencvVecConverter<std::vector<_Tp> >
{
    static bool to(PyObject* obj, std::vector<std::vector<_Tp> >& value, const ArgInfo info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const std::vector<std::vector<_Tp> >& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

template<> struct pyopencvVecConverter<Mat>
{
    static bool to(PyObject* obj, std::vector<Mat>& value, const ArgInfo info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const std::vector<Mat>& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

template<> struct pyopencvVecConverter<UMat>
{
    static bool to(PyObject* obj, std::vector<UMat>& value, const ArgInfo info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const std::vector<UMat>& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

template<> struct pyopencvVecConverter<KeyPoint>
{
    static bool to(PyObject* obj, std::vector<KeyPoint>& value, const ArgInfo info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const std::vector<KeyPoint>& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

template<> struct pyopencvVecConverter<DMatch>
{
    static bool to(PyObject* obj, std::vector<DMatch>& value, const ArgInfo info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const std::vector<DMatch>& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

template<> struct pyopencvVecConverter<String>
{
    static bool to(PyObject* obj, std::vector<String>& value, const ArgInfo info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const std::vector<String>& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

template<> struct pyopencvVecConverter<RotatedRect>
{
    static bool to(PyObject* obj, std::vector<RotatedRect>& value, const ArgInfo info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }
    static PyObject* from(const std::vector<RotatedRect>& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

template<>
bool pyopencv_to(PyObject *obj, TermCriteria& dst, const char *name)
{
    CV_UNUSED(name);
    if(!obj)
        return true;
    return PyArg_ParseTuple(obj, "iid", &dst.type, &dst.maxCount, &dst.epsilon) > 0;
}

template<>
PyObject* pyopencv_from(const TermCriteria& src)
{
    return Py_BuildValue("(iid)", src.type, src.maxCount, src.epsilon);
}

template<>
bool pyopencv_to(PyObject *obj, RotatedRect& dst, const char *name)
{
    CV_UNUSED(name);
    if(!obj)
        return true;
    return PyArg_ParseTuple(obj, "(ff)(ff)f", &dst.center.x, &dst.center.y, &dst.size.width, &dst.size.height, &dst.angle) > 0;
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
static PyObject *pycvCreateTrackbar(PyObject*, PyObject *args)
{
    PyObject *on_change;
    char* trackbar_name;
    char* window_name;
    int *value = new int;
    int count;

    if (!PyArg_ParseTuple(args, "ssiiO", &trackbar_name, &window_name, value, &count, &on_change))
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
    ERRWRAP2(createTrackbar(trackbar_name, window_name, value, count, OnChange, py_callback_info));
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

static int convert_to_char(PyObject *o, char *dst, const char *name = "no_name")
{
    std::string str;
    if (getUnicodeString(o, str))
    {
        *dst = str[0];
        return 1;
    }
    (*dst) = 0;
    return failmsg("Expected single character string for argument '%s'", name);
}

#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wunused-parameter"
#  pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif


#include "pyopencv_generated_enums.h"
#include "pyopencv_custom_headers.h"

#ifdef CVPY_DYNAMIC_INIT
#define CVPY_TYPE(NAME, STORAGE, SNAME, _1, _2) CVPY_TYPE_DECLARE_DYNAMIC(NAME, STORAGE, SNAME)
#else
#define CVPY_TYPE(NAME, STORAGE, SNAME, _1, _2) CVPY_TYPE_DECLARE(NAME, STORAGE, SNAME)
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
#define CVPY_TYPE(NAME, _1, _2, BASE, CONSTRUCTOR) CVPY_TYPE_INIT_DYNAMIC(NAME, return false, BASE, CONSTRUCTOR)
    PyObject * pyopencv_NoBase_TypePtr = NULL;
#else
#define CVPY_TYPE(NAME, _1, _2, BASE, CONSTRUCTOR) CVPY_TYPE_INIT_STATIC(NAME, return false, BASE, CONSTRUCTOR)
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

#if PY_MAJOR_VERSION >= 3
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
