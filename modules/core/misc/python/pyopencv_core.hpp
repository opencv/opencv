#ifdef HAVE_OPENCV_CORE

#include "opencv2/core/cuda.hpp"

typedef std::vector<cuda::GpuMat> vector_GpuMat;
typedef cuda::GpuMat::Allocator GpuMat_Allocator;
typedef cuda::HostMem::AllocType HostMem_AllocType;
typedef cuda::Event::CreateFlags Event_CreateFlags;

CV_PY_TO_CLASS(cuda::GpuMat);
CV_PY_TO_CLASS(cuda::Stream);
CV_PY_TO_CLASS(cuda::Event);
CV_PY_TO_CLASS(cuda::HostMem);

CV_PY_TO_CLASS_PTR(cuda::GpuMat);
CV_PY_TO_CLASS_PTR(cuda::GpuMat::Allocator);

CV_PY_TO_ENUM(cuda::Event::CreateFlags);
CV_PY_TO_ENUM(cuda::HostMem::AllocType);
CV_PY_TO_ENUM(cuda::FeatureSet);

CV_PY_FROM_CLASS(cuda::GpuMat);
CV_PY_FROM_CLASS(cuda::Stream);
CV_PY_FROM_CLASS(cuda::HostMem);

CV_PY_FROM_CLASS_PTR(cuda::GpuMat::Allocator);

CV_PY_FROM_ENUM(cuda::DeviceInfo::ComputeMode);

#endif
