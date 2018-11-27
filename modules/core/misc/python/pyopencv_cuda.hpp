#ifdef HAVE_OPENCV_CORE

#include "opencv2/core/cuda.hpp"

typedef std::vector<cuda::GpuMat> vector_GpuMat;
typedef cuda::GpuMat::Allocator GpuMat_Allocator;
typedef cuda::HostMem::AllocType HostMem_AllocType;
typedef cuda::Event::CreateFlags Event_CreateFlags;

template<> struct pyopencvVecConverter<cuda::GpuMat>
{
    static bool to(PyObject* obj, std::vector<cuda::GpuMat>& value, const ArgInfo info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const std::vector<cuda::GpuMat>& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

CV_PY_TO_CLASS(cuda::GpuMat);
CV_PY_TO_CLASS(cuda::Stream);
CV_PY_TO_CLASS(cuda::Event);
CV_PY_TO_CLASS(cuda::HostMem);

CV_PY_TO_CLASS_PTR(cuda::GpuMat);
CV_PY_TO_CLASS_PTR(cuda::GpuMat::Allocator);

CV_PY_FROM_CLASS(cuda::GpuMat);
CV_PY_FROM_CLASS(cuda::Stream);
CV_PY_FROM_CLASS(cuda::HostMem);

CV_PY_FROM_CLASS_PTR(cuda::GpuMat::Allocator);

#endif
