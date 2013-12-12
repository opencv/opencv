#ifndef __OPENCV_CORE_OCL_RUNTIME_HPP__
#define __OPENCV_CORE_OCL_RUNTIME_HPP__

#ifdef HAVE_OPENCL

#if defined(HAVE_OPENCL_STATIC)

#if defined __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#else // HAVE_OPENCL_STATIC

#include "ocl_runtime_opencl.hpp"

#endif // HAVE_OPENCL_STATIC

#ifndef CL_DEVICE_DOUBLE_FP_CONFIG
#define CL_DEVICE_DOUBLE_FP_CONFIG 0x1032
#endif

#ifndef CL_DEVICE_HALF_FP_CONFIG
#define CL_DEVICE_HALF_FP_CONFIG 0x1033
#endif

#ifndef CL_VERSION_1_2
#define CV_REQUIRE_OPENCL_1_2_ERROR CV_ErrorNoReturn(cv::Error::OpenCLApiCallError, "OpenCV compiled without OpenCL v1.2 support, so we can't use functionality from OpenCL v1.2")
#endif

#endif // HAVE_OPENCL

#endif // __OPENCV_CORE_OCL_RUNTIME_HPP__
