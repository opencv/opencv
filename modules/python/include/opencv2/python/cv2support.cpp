#include "numpy/ndarrayobject.h"

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

ArgInfo::ArgInfo(const char * name_, bool outputarg_)
        : name(name_)
        , outputarg(outputarg_) {}

// to match with older pyopencv_to function signature
// ArgInfo::operator char const*() const { return name; }


PyAllowThreads::PyAllowThreads() : _state(PyEval_SaveThread()) {}
PyAllowThreads::~PyAllowThreads()
{
    PyEval_RestoreThread(_state);
}



PyEnsureGIL::PyEnsureGIL() : _state(PyGILState_Ensure()) {}
PyEnsureGIL::~PyEnsureGIL()
{
    PyGILState_Release(_state);
}



NumpyAllocator::NumpyAllocator() { stdAllocator = cv::Mat::getStdAllocator(); }
NumpyAllocator::~NumpyAllocator() {}

cv::UMatData* NumpyAllocator::allocate(PyObject* o, int dims, const int* sizes, int type, size_t* step) const
{
    cv::UMatData* u = new cv::UMatData(this);
    u->refcount = 1;
    u->data = u->origdata = (uchar*)PyArray_DATA((PyArrayObject*) o);
    npy_intp* _strides = PyArray_STRIDES((PyArrayObject*) o);
    for( int i = 0; i < dims - 1; i++ )
        step[i] = (size_t)_strides[i];
    step[dims-1] = CV_ELEM_SIZE(type);
    u->size = sizes[0]*step[0];
    u->userdata = o;
    return u;
}

cv::UMatData* NumpyAllocator::allocate(int dims0, const int* sizes, int type, size_t* step) const
{
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
    PyObject* o = PyArray_SimpleNew(dims, _sizes, typenum);
    if(!o)
        CV_Error_(cv::Error::StsError, ("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims));
    return allocate(o, dims0, sizes, type, step);
}

bool NumpyAllocator::allocate(cv::UMatData* u, int accessFlags) const
{
    return stdAllocator->allocate(u, accessFlags);
}

void NumpyAllocator::deallocate(cv::UMatData* u) const
{
    if(u)
    {
        PyEnsureGIL gil;
        PyObject* o = (PyObject*)u->userdata;
        Py_DECREF(o);
        delete u;
    }
}

void NumpyAllocator::map(cv::UMatData*, int) const
{

}

void NumpyAllocator::unmap(cv::UMatData* u) const
{
    if(u->urefcount == 0)
        deallocate(u);
}

void NumpyAllocator::download(cv::UMatData* u, void* dstptr,
              int dims, const size_t sz[],
              const size_t srcofs[], const size_t srcstep[],
              const size_t dststep[]) const
{
    stdAllocator->download(u, dstptr, dims, sz, srcofs, srcstep, dststep);
}

void NumpyAllocator::upload(cv::UMatData* u, const void* srcptr, int dims, const size_t sz[],
            const size_t dstofs[], const size_t dststep[],
            const size_t srcstep[]) const
{
    stdAllocator->upload(u, srcptr, dims, sz, dstofs, dststep, srcstep);
}

void NumpyAllocator::copy(cv::UMatData* usrc, cv::UMatData* udst, int dims, const size_t sz[],
          const size_t srcofs[], const size_t srcstep[],
          const size_t dstofs[], const size_t dststep[], bool sync) const
{
    stdAllocator->copy(usrc, udst, dims, sz, srcofs, srcstep, dstofs, dststep, sync);
}

NumpyAllocator g_numpyAllocator;

enum { ARG_NONE = 0, ARG_MAT = 1, ARG_SCALAR = 2 };

// special case, when the convertor needs full ArgInfo structure
template<>
bool pyopencv_to(PyObject* o, cv::Mat& m, const ArgInfo info)
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
        double v[] = {PyInt_AsLong((PyObject*)o), 0., 0., 0.};
        m = cv::Mat(4, 1, CV_64F, v).clone();
        return true;
    }
    if( PyFloat_Check(o) )
    {
        double v[] = {PyFloat_AsDouble((PyObject*)o), 0., 0., 0.};
        m = cv::Mat(4, 1, CV_64F, v).clone();
        return true;
    }
    if( PyTuple_Check(o) )
    {
        int i, sz = (int)PyTuple_Size((PyObject*)o);
        m = cv::Mat(sz, 1, CV_64F);
        for( i = 0; i < sz; i++ )
        {
            PyObject* oi = PyTuple_GET_ITEM(o, i);
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
        if( typenum == NPY_INT64 || typenum == NPY_UINT64 || type == NPY_LONG )
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
        if( (i == ndims-1 && (size_t)_strides[i] != elemsize) ||
            (i < ndims-1 && _strides[i] < _strides[i+1]) )
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

    for(int i = 0; i < ndims; i++)
    {
        size[i] = (int)_sizes[i];
        step[i] = (size_t)_strides[i];
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

    m = cv::Mat(ndims, size, type, PyArray_DATA(oarr), step);
    m.u = g_numpyAllocator.allocate(o, ndims, size, type, step);

    if( !needcopy )
    {
        Py_INCREF(o);
    };
    m.allocator = &g_numpyAllocator;

    return true;
}

template<>
PyObject* pyopencv_from(const cv::Mat& m)
{
    if( !m.data )
        Py_RETURN_NONE;
    cv::Mat temp, *p = (cv::Mat*)&m;
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

bool pyopencv_coerce(PyObject *o, cv::Scalar& s, const ArgInfo info)
{
    if(!o || o == Py_None)
        return true;
    if (PySequence_Check(o)) {
        PyObject *fi = PySequence_Fast(o, info.name);
        if (fi == NULL)
            return false;
        if (4 < PySequence_Fast_GET_SIZE(fi))
        {
            failmsg("Scalar value for argument '%s' is longer than 4", info.name);
            return false;
        }
        for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fi); i++) {
            PyObject *item = PySequence_Fast_GET_ITEM(fi, i);
            if (PyFloat_Check(item) || PyInt_Check(item)) {
                s[(int)i] = PyFloat_AsDouble(item);
            } else {
                failmsg("Scalar value for argument '%s' is not numeric", info.name);
                return false;
            }
        }
        Py_DECREF(fi);
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
bool pyopencv_to(PyObject *obj, cv::Scalar& value, const ArgInfo info)
{
    return pyopencv_coerce(obj, value, info);
}

template<>
PyObject* pyopencv_from(const cv::Scalar& src)
{
    return Py_BuildValue("(dddd)", src[0], src[1], src[2], src[3]);
}


template<>
PyObject* pyopencv_from(const bool& value)
{
    return PyBool_FromLong(value);
}

bool pyopencv_coerce(PyObject* obj, bool& value)
{
    if(!obj || obj == Py_None)
        return true;
    int _val = PyObject_IsTrue(obj);
    if(_val < 0)
        return false;
    value = _val > 0;
    return true;
}

template<>
bool pyopencv_to(PyObject* obj, bool& value, const ArgInfo info)
{
    (void)info.name;
    return pyopencv_coerce(obj, value);
}

template<>
PyObject* pyopencv_from(const size_t& value)
{
    return PyLong_FromSize_t(value);
}

bool pyopencv_coerce(PyObject* obj, size_t& value)
{
    if(!obj || obj == Py_None)
        return true;
    value = (int)PyLong_AsUnsignedLong(obj);
    return value != (size_t)-1 || !PyErr_Occurred();
}

template<>
bool pyopencv_to(PyObject* obj, size_t& value, const ArgInfo info)
{
    (void)info.name;
    return pyopencv_coerce(obj, value);
}

template<>
PyObject* pyopencv_from(const int& value)
{
    return PyInt_FromLong(value);
}

bool pyopencv_coerce(PyObject* obj, int& value) {
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

template<>
bool pyopencv_to(PyObject* obj, int& value, const ArgInfo info)
{
    (void)info.name;
    return pyopencv_coerce(obj, value);
}


template<>
PyObject* pyopencv_from(const int64& value)
{
    return PyLong_FromLongLong(value);
}

template<>
PyObject* pyopencv_from(const uchar& value)
{
    return PyInt_FromLong(value);
}

bool pyopencv_coerce(PyObject* obj, uchar& value)
{
    if(!obj || obj == Py_None)
        return true;
    int ivalue = (int)PyInt_AsLong(obj);
    value = cv::saturate_cast<uchar>(ivalue);
    return ivalue != -1 || !PyErr_Occurred();
}

template<>
bool pyopencv_to(PyObject* obj, uchar& value, const ArgInfo info)
{
    (void)info.name;
    return pyopencv_coerce(obj, value);
}

template<>
PyObject* pyopencv_from(const double& value)
{
    return PyFloat_FromDouble(value);
}

bool pyopencv_coerce(PyObject* obj, double& value)
{
    if(!obj || obj == Py_None)
        return true;
    if(!!PyInt_CheckExact(obj))
        value = (double)PyInt_AS_LONG(obj);
    else
        value = PyFloat_AsDouble(obj);
    return !PyErr_Occurred();
}

template<>
bool pyopencv_to(PyObject* obj, double& value, const ArgInfo info)
{
    (void)info.name;
    return pyopencv_coerce(obj, value);
}

template<>
PyObject* pyopencv_from(const float& value)
{
    return PyFloat_FromDouble(value);
}

bool pyopencv_coerce(PyObject* obj, float& value)
{
    if(!obj || obj == Py_None)
        return true;
    if(!!PyInt_CheckExact(obj))
        value = (float)PyInt_AS_LONG(obj);
    else
        value = (float)PyFloat_AsDouble(obj);
    return !PyErr_Occurred();
}

template<>
bool pyopencv_to(PyObject* obj, float& value, const ArgInfo info)
{
    (void)info.name;
    return pyopencv_coerce(obj, value);
}

template<>
PyObject* pyopencv_from(const std::string& value)
{
    return PyString_FromString(value.empty() ? "" : value.c_str());
}

bool pyopencv_coerce(PyObject* obj, std::string& value)
{
    if(!obj || obj == Py_None)
        return true;
    char* str = PyString_AsString(obj);
    if(!str)
        return false;
    value = std::string(str);
    return true;
}

template<>
bool pyopencv_to(PyObject* obj, std::string& value, const ArgInfo info)
{
    (void)info.name;
    return pyopencv_coerce(obj, value);
}

template<>
PyObject* pyopencv_from(const cv::String& value)
{
    return PyString_FromString(value.empty() ? "" : value.c_str());
}

bool pyopencv_coerce(PyObject* obj, cv::String& value)
{
    if(!obj || obj == Py_None)
        return true;
    char* str = PyString_AsString(obj);
    if(!str)
        return false;
    value = cv::String(str);
    return true;
}

template<>
bool pyopencv_to(PyObject* obj, cv::String& value, const ArgInfo info)
{
    (void)info.name;
    return pyopencv_coerce(obj, value);
}

template<>
PyObject* pyopencv_from(const cv::Rect& r)
{
    return Py_BuildValue("(iiii)", r.x, r.y, r.width, r.height);
}

template<>
bool pyopencv_to(PyObject* obj, cv::Rect& r, const ArgInfo info)
{
    (void)info.name;
    if(!obj || obj == Py_None)
        return true;
    return PyArg_ParseTuple(obj, "iiii", &r.x, &r.y, &r.width, &r.height) > 0;
}

template<>
PyObject* pyopencv_from(const cv::Point& p)
{
    return Py_BuildValue("(ii)", p.x, p.y);
}

template<>
bool pyopencv_to(PyObject* obj, cv::Point& p, const ArgInfo info)
{
    (void)info.name;
    if(!obj || obj == Py_None)
        return true;
    if(!!PyComplex_CheckExact(obj))
    {
        Py_complex c = PyComplex_AsCComplex(obj);
        p.x = cv::saturate_cast<int>(c.real);
        p.y = cv::saturate_cast<int>(c.imag);
        return true;
    }
    return PyArg_ParseTuple(obj, "ii", &p.x, &p.y) > 0;
}

template<>
PyObject* pyopencv_from(const cv::Point2f& p)
{
    return Py_BuildValue("(dd)", p.x, p.y);
}

bool pyopencv_coerce(PyObject* obj, cv::Point2f& p)
{
    if(!obj || obj == Py_None)
        return true;
    if(!!PyComplex_CheckExact(obj))
    {
        Py_complex c = PyComplex_AsCComplex(obj);
        p.x = cv::saturate_cast<float>(c.real);
        p.y = cv::saturate_cast<float>(c.imag);
        return true;
    }
    return PyArg_ParseTuple(obj, "ff", &p.x, &p.y) > 0;
}

template<>
bool pyopencv_to(PyObject* obj, cv::Point2f& value, const ArgInfo info)
{
    (void)info.name;
    return pyopencv_coerce(obj, value);
}

template<>
PyObject* pyopencv_from(const cv::Vec2d& v)
{
    return Py_BuildValue("(dd)", v[0], v[1]);
}

template<>
PyObject* pyopencv_from(const cv::Vec3d& v)
{
    return Py_BuildValue("(ddd)", v[0], v[1], v[2]);
}

template<>
bool pyopencv_to(PyObject* obj, cv::Vec3d& v, const ArgInfo info)
{
    (void)info.name;
    if(!obj)
        return true;
    return PyArg_ParseTuple(obj, "ddd", &v[0], &v[1], &v[2]) > 0;
}

template<>
PyObject* pyopencv_from(const cv::Point2d& p)
{
    return Py_BuildValue("(dd)", p.x, p.y);
}

template<>
bool pyopencv_to(PyObject* obj, cv::Point2d& p, const ArgInfo info)
{
    (void)info.name;
    if(!obj || obj == Py_None)
        return true;
    if(!!PyComplex_CheckExact(obj))
    {
        Py_complex c = PyComplex_AsCComplex(obj);
        p.x = cv::saturate_cast<int>(c.real);
        p.y = cv::saturate_cast<int>(c.imag);
        return true;
    }
    return PyArg_ParseTuple(obj, "ii", &p.x, &p.y) > 0;
}

template<>
PyObject* pyopencv_from(const cv::RotatedRect& src)
{
    return Py_BuildValue("((ff)(ff)f)", src.center.x, src.center.y, src.size.width, src.size.height, src.angle);
}

template<>
bool pyopencv_to(PyObject *obj, cv::RotatedRect& dst, const ArgInfo info)
{
    (void)info.name;
    if(!obj)
        return true;
    return PyArg_ParseTuple(obj, "(ff)(ff)f", &dst.center.x, &dst.center.y, &dst.size.width, &dst.size.height, &dst.angle) > 0;
}

bool pyopencv_coerce(PyObject* obj, cv::Size& sz)
{
    if(!obj || obj == Py_None)
        return true;
    return PyArg_ParseTuple(obj, "ii", &sz.width, &sz.height) > 0;
}

template<>
bool pyopencv_to(PyObject* obj, cv::Size& sz, const ArgInfo info)
{
    (void)info.name;
    return pyopencv_coerce(obj, sz);
}

template<>
PyObject* pyopencv_from(const cv::Size& sz)
{
    return Py_BuildValue("(ii)", sz.width, sz.height);
}


template<>
PyObject* pyopencv_from(const cv::Moments& m)
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

template<typename _Tp> struct pyopencvVecConverter
{
    static bool to(PyObject* obj, std::vector<_Tp>& value, const ArgInfo info)
    {
        typedef typename cv::DataType<_Tp>::channel_type _Cp;
        if(!obj || obj == Py_None)
            return true;
        if (PyArray_Check(obj))
        {
            cv::Mat m;
            pyopencv_to(obj, m, info);
            m.copyTo(value);
        }
        if (!PySequence_Check(obj))
            return false;
        PyObject *seq = PySequence_Fast(obj, info.name);
        if (seq == NULL)
            return false;
        int i, j, n = (int)PySequence_Fast_GET_SIZE(seq);
        value.resize(n);

        int type = cv::DataType<_Tp>::type;
        int depth = CV_MAT_DEPTH(type), channels = CV_MAT_CN(type);
        PyObject** items = PySequence_Fast_ITEMS(seq);

        for( i = 0; i < n; i++ )
        {
            PyObject* item = items[i];
            PyObject* seq_i = 0;
            PyObject** items_i = &item;
            _Cp* data = (_Cp*)&value[i];

            if( channels == 2 && PyComplex_CheckExact(item) )
            {
                Py_complex c = PyComplex_AsCComplex(obj);
                data[0] = cv::saturate_cast<_Cp>(c.real);
                data[1] = cv::saturate_cast<_Cp>(c.imag);
                continue;
            }
            if( channels > 1 )
            {
                if( PyArray_Check(item))
                {
                    cv::Mat src;
                    pyopencv_to(item, src, info);
                    if( src.dims != 2 || src.channels() != 1 ||
                       ((src.cols != 1 || src.rows != channels) &&
                        (src.cols != channels || src.rows != 1)))
                        break;
                    cv::Mat dst(src.rows, src.cols, depth, data);
                    src.convertTo(dst, type);
                    if( dst.data != (uchar*)data )
                        break;
                    continue;
                }

                seq_i = PySequence_Fast(item, info.name);
                if( !seq_i || (int)PySequence_Fast_GET_SIZE(seq_i) != channels )
                {
                    Py_XDECREF(seq_i);
                    break;
                }
                items_i = PySequence_Fast_ITEMS(seq_i);
            }

            for( j = 0; j < channels; j++ )
            {
                PyObject* item_ij = items_i[j];
                if( PyInt_Check(item_ij))
                {
                    int v = (int)PyInt_AsLong(item_ij);
                    if( v == -1 && PyErr_Occurred() )
                        break;
                    data[j] = cv::saturate_cast<_Cp>(v);
                }
                else if( PyLong_Check(item_ij))
                {
                    int v = (int)PyLong_AsLong(item_ij);
                    if( v == -1 && PyErr_Occurred() )
                        break;
                    data[j] = cv::saturate_cast<_Cp>(v);
                }
                else if( PyFloat_Check(item_ij))
                {
                    double v = PyFloat_AsDouble(item_ij);
                    if( PyErr_Occurred() )
                        break;
                    data[j] = cv::saturate_cast<_Cp>(v);
                }
                else
                    break;
            }
            Py_XDECREF(seq_i);
            if( j < channels )
                break;
        }
        Py_DECREF(seq);
        return i == n;
    }

    static PyObject* from(const std::vector<_Tp>& value)
    {
        if(value.empty())
            return PyTuple_New(0);
        cv::Mat src((int)value.size(), cv::DataType<_Tp>::channels, cv::DataType<_Tp>::depth, (uchar*)&value[0]);
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
    PyObject *seq = PySequence_Fast(obj, info.name);
    if (seq == NULL)
        return false;
    int i, n = (int)PySequence_Fast_GET_SIZE(seq);
    value.resize(n);

    PyObject** items = PySequence_Fast_ITEMS(seq);

    for( i = 0; i < n; i++ )
    {
        PyObject* item = items[i];
        if(!pyopencv_to(item, value[i], info))
            break;
    }
    Py_DECREF(seq);
    return i == n;
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
        PyList_SET_ITEM(seq, i, item);
    }
    if( i < n )
    {
        Py_DECREF(seq);
        return 0;
    }
    return seq;
}


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

template<> struct pyopencvVecConverter<cv::Mat>
{
    static bool to(PyObject* obj, std::vector<cv::Mat>& value, const ArgInfo info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const std::vector<cv::Mat>& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

template<> struct pyopencvVecConverter<cv::String>
{
    static bool to(PyObject* obj, std::vector<cv::String>& value, const ArgInfo info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const std::vector<cv::String>& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

#if PY_MAJOR_VERSION >= 3
#define MKTYPE2(NAME) pyopencv_##NAME##_specials(); if (!to_ok(&pyopencv_##NAME##_Type)) return NULL;
#else
#define MKTYPE2(NAME) pyopencv_##NAME##_specials(); if (!to_ok(&pyopencv_##NAME##_Type)) return;
#endif

#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wunused-parameter"
#  pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif

static int to_ok(PyTypeObject *to)
{
  to->tp_alloc = PyType_GenericAlloc;
  to->tp_new = PyType_GenericNew;
  to->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  return (PyType_Ready(to) == 0);
}
