#ifndef OPENCV2X_PYTHON_WRAPPERS
#define OPENCV2X_PYTHON_WRAPPERS

#include "opencv2/core/core.hpp"

namespace cv
{

#define ERRWRAP2(expr) \
try \
{ \
    expr; \
} \
catch (const cv::Exception &e) \
{ \
    PyErr_SetString(opencv_error, e.what()); \
    return 0; \
}

static size_t REFCOUNT_OFFSET = (size_t)&(((PyObject*)0)->ob_refcnt) +
    (0x12345678 != *(const size_t*)"\x78\x56\x34\x12\0\0\0\0\0")*sizeof(int);

static inline PyObject* pyObjectFromRefcount(const int* refcount)
{
    return (PyObject*)((size_t)refcount - REFCOUNT_OFFSET);
}

static inline int* refcountFromPyObject(const PyObject* obj)
{
    return (int*)((size_t)obj + REFCOUNT_OFFSET);
}

class NumpyAllocator : public MatAllocator
{
public:
    NumpyAllocator() {}
    ~NumpyAllocator() {}
    
    void allocate(int dims, const int* sizes, int type, int*& refcount,
                  uchar*& datastart, uchar*& data, size_t* step)
    {
        static int ncalls = 0;
        printf("NumpyAllocator::allocate: %d\n", ncalls++);
        
        int depth = CV_MAT_DEPTH(type);
        int cn = CV_MAT_CN(type);
        const int f = (int)(sizeof(size_t)/8);
        int typenum = depth == CV_8U ? NPY_UBYTE : depth == CV_8S ? NPY_BYTE :
                      depth == CV_16U ? NPY_USHORT : depth == CV_16S ? NPY_SHORT :
                      depth == CV_32S ? NPY_INT : depth == CV_32F ? NPY_FLOAT :
                      depth == CV_64F ? NPY_DOUBLE : f*NPY_ULONGLONG + (f^1)*NPY_UINT;
        int i;
        npy_intp _sizes[CV_MAX_DIM+1];
        for( i = 0; i < dims; i++ )
            _sizes[i] = sizes[i];
        if( cn > 1 )
        {
            if( _sizes[dims-1] == 1 )
                _sizes[dims-1] = cn;
            else
                _sizes[dims++] = cn;
        }
        PyObject* o = PyArray_SimpleNew(dims, _sizes, typenum);
        if(!o)
            CV_Error_(CV_StsError, ("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims));
        refcount = refcountFromPyObject(o);
        npy_intp* _strides = PyArray_STRIDES(o);
        for( i = 0; i < dims-1; i++ )
            step[i] = (size_t)_strides[i];
        datastart = data = (uchar*)PyArray_DATA(o);
    }
    
    void deallocate(int* refcount, uchar* datastart, uchar* data)
    {
        static int ncalls = 0;
        printf("NumpyAllocator::deallocate: %d\n", ncalls++);
        
        if( !refcount )
            return;
        PyObject* o = pyObjectFromRefcount(refcount);
        Py_DECREF(o);
    }
};

NumpyAllocator g_numpyAllocator;
    
enum { ARG_NONE = 0, ARG_MAT = 1, ARG_SCALAR = 2 };

static int pyobjToMat(const PyObject* o, Mat& m, const char* name = "<unknown>", bool allowND=true)
{
    static int call_idx = 0;
    printf("pyobjToMatND: %d\n", call_idx++);
    
    if( !PyArray_Check(o) ) {
        if( o == Py_None )
            return ARG_NONE;
        failmsg("%s is not a numpy array", name);
        return -1;
    }
    
    int typenum = PyArray_TYPE(o);
    int type = typenum == NPY_UBYTE ? CV_8U : typenum == NPY_BYTE ? CV_8S :
               typenum == NPY_USHORT ? CV_16U : typenum == NPY_SHORT ? CV_16S : 
               typenum == NPY_INT || typenum == NPY_LONG ? CV_32S :
               typenum == NPY_FLOAT ? CV_32F :
               typenum == NPY_DOUBLE ? CV_64F : -1;
    
    if( type < 0 )
    {
        failmsg("%s data type = %d is not supported", name, typenum);
        return -1;
    }
    
    int ndims = PyArray_NDIM(o);
    if(ndims >= CV_MAX_DIM)
    {
        failmsg("%s dimensionality (=%d) is too high", name, ndims);
        return -1;
    }
    
    int size[CV_MAX_DIM+1];
    size_t step[CV_MAX_DIM+1], elemsize = CV_ELEM_SIZE(type);
    const npy_intp* _sizes = PyArray_DIMS(o);
    const npy_intp* _strides = PyArray_STRIDES(o);
    
    for(int i = 0; i < ndims; i++)
    {
        size[i] = (int)_sizes[i];
        step[i] = (size_t)_strides[i];
    }
    
    if( ndims == 0 || step[ndims-1] > elemsize ) {
        size[ndims] = 1;
        step[ndims] = elemsize;
        ndims++;
    }
    
    m = Mat(ndims, size, type, PyArray_DATA(o), step);
    
    if (!allowND)
    {
        if( ndims <= 2 )
            ;
        else if( ndims == 3 )
        {
            if( size[2] > CV_CN_MAX || step[1] != elemsize*size[2] )
            {
                failmsg("%s is not contiguous, thus it can not be interpreted as image", name);
                return -1;
            }
            m.dims--;
            m.flags = (m.flags & ~CV_MAT_TYPE_MASK) | CV_MAKETYPE(type, size[2]);
        }
        else
        {
            failmsg("%s is not contiguous or has more than 3 dimensions, thus it can not be interpreted as image", name);
            return -1;
        }
    }
    
    if( m.data )
    {
        m.refcount = refcountFromPyObject(o);
        ++*m.refcount; // protect the original numpy array from deallocation
                       // (since Mat destructor will decrement the reference counter)
    };
    m.allocator = &g_numpyAllocator;
    return ARG_MAT;
}

static void makeEmptyMat(Mat& m)
{
    m = Mat();
    m.allocator = &g_numpyAllocator;
}

static int pyobjToMat(const PyObject* o, Mat& m, const char* name = "<unknown>")
{
    Mat temp;
    int code = pyobjToMat(o, temp, name, false);
    if(code > 0)
        m = Mat(temp);
    return code;
}

static int pyobjToScalar(PyObject *o, Scalar& s, const char *name = "<unknown>")
{
    if (PySequence_Check(o)) {
        PyObject *fi = PySequence_Fast(o, name);
        if (fi == NULL)
            return -1;
        if (4 < PySequence_Fast_GET_SIZE(fi))
        {
            failmsg("Scalar value for argument '%s' is longer than 4", name);
            return -1;
        }
        for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fi); i++) {
            PyObject *item = PySequence_Fast_GET_ITEM(fi, i);
            if (PyFloat_Check(item) || PyInt_Check(item)) {
                s[i] = PyFloat_AsDouble(item);
            } else {
                failmsg("Scalar value for argument '%s' is not numeric", name);
                return -1;
            }
        }
        Py_DECREF(fi);
    } else {
        if (PyFloat_Check(o) || PyInt_Check(o)) {
            s[0] = PyFloat_AsDouble(o);
        } else {
            failmsg("Scalar value for argument '%s' is not numeric", name);
            return -1;
        }
    }
    return ARG_SCALAR;
}


static int pyobjToMatOrScalar(PyObject* obj, Mat& m, Scalar& s, const char* name, bool allowND)
{
    if( PyArray_Check(obj) || (obj == Py_None))
        return pyobjToMat(obj, m, name, allowND);

    return pyobjToScalar(obj, s, name);
}

static void pyc_add_mm(const Mat& a, const Mat& b, Mat& c, const Mat& mask) { add(a, b, c, mask); }
static void pyc_add_ms(const Mat& a, const Scalar& s, Mat& c, const Mat& mask, bool) { add(a, s, c, mask); }
static void pyc_subtract_mm(const Mat& a, const Mat& b, Mat& c, const Mat& mask) { subtract(a, b, c, mask); }
static void pyc_subtract_ms(const Mat& a, const Scalar& s, Mat& c, const Mat& mask, bool rev)
{
    if( !rev )
        subtract(a, s, c, mask);
    else
        subtract(s, a, c, mask);
}

static void pyc_and_mm(const Mat& a, const Mat& b, Mat& c, const Mat& mask) { bitwise_and(a, b, c, mask); }
static void pyc_and_ms(const Mat& a, const Scalar& s, Mat& c, const Mat& mask, bool) { bitwise_and(a, s, c, mask); }
static void pyc_or_mm(const Mat& a, const Mat& b, Mat& c, const Mat& mask) { bitwise_or(a, b, c, mask); }
static void pyc_or_ms(const Mat& a, const Scalar& s, Mat& c, const Mat& mask, bool) { bitwise_or(a, s, c, mask); }
static void pyc_xor_mm(const Mat& a, const Mat& b, Mat& c, const Mat& mask) { bitwise_xor(a, b, c, mask); }
static void pyc_xor_ms(const Mat& a, const Scalar& s, Mat& c, const Mat& mask, bool) { bitwise_xor(a, s, c, mask); }
static void pyc_absdiff_mm(const Mat& a, const Mat& b, Mat& c, const Mat&) { absdiff(a, b, c); }
static void pyc_absdiff_ms(const Mat& a, const Scalar& s, Mat& c, const Mat&, bool) { absdiff(a, s, c); }
static void pyc_min_mm(const Mat& a, const Mat& b, Mat& c, const Mat&) { min(a, b, c); }
static void pyc_min_ms(const Mat& a, const Scalar& s, Mat& c, const Mat&, bool)
{
    CV_Assert( s.isReal() );
    min(a, s[0], c);
}
static void pyc_max_mm(const Mat& a, const Mat& b, Mat& c, const Mat&) { max(a, b, c); }
static void pyc_max_ms(const Mat& a, const Scalar& s, Mat& c, const Mat&, bool)
{
    CV_Assert( s.isReal() );
    max(a, s[0], c);
}

typedef void (*BinaryOp)(const Mat& a, const Mat& b, Mat& c, const Mat& mask);
typedef void (*BinaryOpS)(const Mat& a, const Scalar& s, Mat& c, const Mat& mask, bool rev);

static PyObject *pyopencv_binary_op(PyObject* args, PyObject* kw, BinaryOp binOp, BinaryOpS binOpS, bool supportMask)
{
    PyObject *pysrc1 = 0, *pysrc2 = 0, *pydst = 0, *pymask = 0;
    Mat src1, src2, dst, mask;
    Scalar alpha, beta;
    
    if( supportMask )
    {
        const char *keywords[] = { "src1", "src2", "dst", "mask", 0 };
        if( !PyArg_ParseTupleAndKeywords(args, kw, "OO|OO", (char**)keywords,
                                        &pysrc1, &pysrc2, &pydst, &pymask))
            return 0;
    }
    else
    {
        const char *keywords[] = { "src1", "src2", "dst", 0 };
        if( !PyArg_ParseTupleAndKeywords(args, kw, "OO|O", (char**)keywords,
                                         &pysrc1, &pysrc2, &pydst))
            return 0;
    }
    
    int code_src1 = pysrc1 ? pyobjToMatOrScalar(pysrc1, src1, alpha, "src1", true) : -1;
    int code_src2 = pysrc2 ? pyobjToMatOrScalar(pysrc2, src2, beta, "src2", true) : -1;
    int code_dst = pydst ? pyobjToMat(pydst, dst, "dst", true) : 0;
    int code_mask = pymask ? pyobjToMat(pymask, mask, "mask", true) : 0;
    
    if( code_src1 < 0 || code_src2 < 0 || code_dst < 0 || code_mask < 0 )
        return 0;
    
    if( code_src1 == ARG_SCALAR && code_src2 == ARG_SCALAR )
    {
        failmsg("Both %s and %s are scalars", "src1", "src2");
        return 0;
    }
    
    if( code_dst == 0 )
        makeEmptyMat(dst);
    
    ERRWRAP2(code_src1 != ARG_SCALAR && code_src2 != ARG_SCALAR ? binOp(src1, src2, dst, mask) :
             code_src1 != ARG_SCALAR ? binOpS(src2, alpha, dst, mask, true) : binOpS(src1, beta, dst, mask, false));
    
    PyObject* result = pyObjectFromRefcount(dst.refcount);
    int D = PyArray_NDIM(result);
    const npy_intp* sz = PyArray_DIMS(result);
    
    printf("Result: check = %d, ndims = %d, size = (%d x %d), typenum=%d\n", PyArray_Check(result),
           D, (int)sz[0], D >= 2 ? (int)sz[1] : 1, PyArray_TYPE(result));
    
    //Py_INCREF(result);
    return result;
}

static PyObject* pyopencv_add(PyObject* self, PyObject* args, PyObject* kw)
{
    return pyopencv_binary_op(args, kw, pyc_add_mm, pyc_add_ms, true);
}

static PyObject* pyopencv_subtract(PyObject* self, PyObject* args, PyObject* kw)
{
    return pyopencv_binary_op(args, kw, pyc_subtract_mm, pyc_subtract_ms, true);
}

static PyObject* pyopencv_and(PyObject* self, PyObject* args, PyObject* kw)
{
    return pyopencv_binary_op(args, kw, pyc_and_mm, pyc_and_ms, true);
}

static PyObject* pyopencv_or(PyObject* self, PyObject* args, PyObject* kw)
{
    return pyopencv_binary_op(args, kw, pyc_or_mm, pyc_or_ms, true);
}

static PyObject* pyopencv_xor(PyObject* self, PyObject* args, PyObject* kw)
{
    return pyopencv_binary_op(args, kw, pyc_xor_mm, pyc_xor_ms, true);
}

static PyObject* pyopencv_absdiff(PyObject* self, PyObject* args, PyObject* kw)
{
    return pyopencv_binary_op(args, kw, pyc_absdiff_mm, pyc_absdiff_ms, true);
}

static PyObject* pyopencv_min(PyObject* self, PyObject* args, PyObject* kw)
{
    return pyopencv_binary_op(args, kw, pyc_min_mm, pyc_min_ms, true);
}

static PyObject* pyopencv_max(PyObject* self, PyObject* args, PyObject* kw)
{
    return pyopencv_binary_op(args, kw, pyc_max_mm, pyc_max_ms, true);
}

}
    
#endif
