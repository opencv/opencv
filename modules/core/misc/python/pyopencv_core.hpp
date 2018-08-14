#ifdef HAVE_OPENCV_CORE

#include "opencv2/core/cuda.hpp"

typedef std::vector<cuda::GpuMat> vector_GpuMat;
typedef cuda::GpuMat::Allocator GpuMat_Allocator;

template<> bool pyopencv_to(PyObject* o, Ptr<cuda::GpuMat>& m, const char* name);
template<> PyObject* pyopencv_from(const Ptr<cuda::GpuMat>& m);

template<>
bool pyopencv_to(PyObject* o, cuda::GpuMat& m, const char* name)
{
    if (!o || o == Py_None)
        return true;
    Ptr<cuda::GpuMat> mPtr(new cuda::GpuMat());

    if (!pyopencv_to(o, mPtr, name)) return false;
    m = *mPtr;
    return true;
}

template<>
PyObject* pyopencv_from(const cuda::GpuMat& m)
{
    Ptr<cuda::GpuMat> mPtr(new cuda::GpuMat());

    *mPtr = m;
    return pyopencv_from(mPtr);
}

template<>
bool pyopencv_to(PyObject *o, cuda::GpuMat::Allocator* &allocator, const char *name)
{
    (void)name;
    if (!o || o == Py_None)
        return true;

    failmsg("Python binding for cv::cuda::GpuMat::Allocator is not implemented yet.");
    return false;
}

template<>
bool pyopencv_to(PyObject *o, cuda::Stream &stream, const char *name)
{
    (void)name;
    if (!o || o == Py_None)
        return true;

    failmsg("Python binding for cv::cuda::Stream is not implemented yet.");
    return false;
}

#endif
