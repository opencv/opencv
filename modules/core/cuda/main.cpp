#include "opencv2/core/core.hpp"
#include "opencv2/core/gpumat.hpp"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <npp.h>

#define CUDART_MINIMUM_REQUIRED_VERSION 4020
#define NPP_MINIMUM_REQUIRED_VERSION 4200

#if (CUDART_VERSION < CUDART_MINIMUM_REQUIRED_VERSION)
#error "Insufficient Cuda Runtime library version, please update it."
#endif

#if (NPP_VERSION_MAJOR * 1000 + NPP_VERSION_MINOR * 100 + NPP_VERSION_BUILD < NPP_MINIMUM_REQUIRED_VERSION)
#error "Insufficient NPP version, please update it."
#endif
#endif

using namespace cv;
using namespace cv::gpu;

#include "gpumat_cuda.hpp"