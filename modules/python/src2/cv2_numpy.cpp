// must be defined before importing numpy headers
// https://numpy.org/doc/1.17/reference/c-api.array.html#importing-the-api
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL opencv_ARRAY_API

#include "cv2_numpy.hpp"
#include "cv2_util.hpp"

using namespace cv;

UMatData* NumpyAllocator::allocate(PyObject* o, int dims, const int* sizes, int type, size_t* step) const
{
    UMatData* u = new UMatData(this);
    u->data = u->origdata = (uchar*)PyArray_DATA((PyArrayObject*) o);
    npy_intp* _strides = PyArray_STRIDES((PyArrayObject*) o);
    for( int i = 0; i < dims - 1; i++ )
        step[i] = (size_t)_strides[i];
    if( dims > 0 )
        step[dims-1] = CV_ELEM_SIZE(type);
    u->size = dims > 0 ? sizes[0]*step[0] : CV_ELEM_SIZE(type);
    u->userdata = o;
    return u;
}

UMatData* NumpyAllocator::allocate(int dims0, const int* sizes, int type, void* data, size_t* step, AccessFlag flags, UMatUsageFlags usageFlags) const
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
    int i, dims = dims0;
    cv::AutoBuffer<npy_intp> _sizes(dims + 1);
    for( i = 0; i < dims; i++ )
        _sizes[i] = sizes[i];
    if( cn > 1 )
        _sizes[dims++] = cn;

    // bfloat16 / float8 need their ml_dtypes descriptor: cvDepthToNumpyType()'s float32
    // fallback would pair a narrow element step with wide NumPy strides and corrupt the buffer.
    PyArray_Descr* exotic = mlDtypeDescrForCvDepth(depth);
    if( exotic )
    {
        Py_INCREF(exotic);  // PyArray_NewFromDescr steals the reference
        PyObject* eo = PyArray_NewFromDescr(&PyArray_Type, exotic, dims, _sizes.data(),
                                            nullptr, nullptr, 0, nullptr);
        if(!eo)
            CV_Error_(Error::StsError, ("The numpy array of depth=%d, ndims=%d can not be created", depth, dims));
        return allocate(eo, dims0, sizes, type, step);
    }
    if( depth == CV_16BF || depth == CV_8F_E4M3FN || depth == CV_8F_E4M3FNUZ )
        CV_Error(Error::StsNotImplemented,
                 "bfloat16 / float8 arrays cannot be allocated through the NumPy allocator: "
                 "NumPy has no such dtype and the ml_dtypes package is not installed");

    int typenum = cvDepthToNumpyType(depth);
    PyObject* o = PyArray_SimpleNew(dims, _sizes.data(), typenum);
    if(!o)
        CV_Error_(Error::StsError, ("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims));
    return allocate(o, dims0, sizes, type, step);
}

bool NumpyAllocator::allocate(UMatData* u, AccessFlag accessFlags, UMatUsageFlags usageFlags) const
{
    return stdAllocator->allocate(u, accessFlags, usageFlags);
}

void NumpyAllocator::deallocate(UMatData* u) const
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
