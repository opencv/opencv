#include "cvconfig.h"
#include "opencv2/core/core.hpp"
#include "opencv2/core/gpumat.hpp"

#include <stdio.h>
#include <iostream>

#ifdef HAVE_CUDA
# include <cuda_runtime.h>
# include <npp.h>

# define CUDART_MINIMUM_REQUIRED_VERSION 4020
# define NPP_MINIMUM_REQUIRED_VERSION 4200

# if (CUDART_VERSION < CUDART_MINIMUM_REQUIRED_VERSION)
#  error "Insufficient Cuda Runtime library version, please update it."
# endif

# if (NPP_VERSION_MAJOR * 1000 + NPP_VERSION_MINOR * 100 + NPP_VERSION_BUILD < NPP_MINIMUM_REQUIRED_VERSION)
#  error "Insufficient NPP version, please update it."
# endif
#endif

using namespace std;
using namespace cv;
using namespace cv::gpu;

#define throw_nogpu CV_Error(CV_GpuNotSupported, "The library is compiled without CUDA support")

#include "opencv2/dynamicuda/dynamicuda.hpp"

#ifdef HAVE_CUDA
static CudaDeviceInfoFuncTable deviceInfoTable;
static CudaFuncTable gpuTable;
#else
static EmptyDeviceInfoFuncTable deviceInfoTable;
static EmptyFuncTable gpuTable;
#endif

extern "C" {

DeviceInfoFuncTable* deviceInfoFactory();
GpuFuncTable* gpuFactory();

DeviceInfoFuncTable* deviceInfoFactory()
{
    return (DeviceInfoFuncTable*)&deviceInfoTable;
}

GpuFuncTable* gpuFactory()
{
    return (GpuFuncTable*)&gpuTable;
}

}
