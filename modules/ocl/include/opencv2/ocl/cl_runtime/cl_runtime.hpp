#ifndef __OPENCV_OCL_CL_RUNTIME_HPP__
#define __OPENCV_OCL_CL_RUNTIME_HPP__

#ifdef HAVE_OPENCL

#if defined(HAVE_OPENCL_STATIC)

#if defined __APPLE__
// APPLE ignores CL_USE_DEPRECATED_OPENCL_1_1_APIS so use this hack:
#include <OpenCL/cl_platform.h>
#ifdef CL_EXT_PREFIX__VERSION_1_1_DEPRECATED
#undef CL_EXT_PREFIX__VERSION_1_1_DEPRECATED
#define CL_EXT_PREFIX__VERSION_1_1_DEPRECATED
#endif
#ifdef CL_EXT_SUFFIX__VERSION_1_1_DEPRECATED
#undef CL_EXT_SUFFIX__VERSION_1_1_DEPRECATED
#define CL_EXT_SUFFIX__VERSION_1_1_DEPRECATED
#endif

#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#else // HAVE_OPENCL_STATIC

#include "cl_runtime_opencl.hpp"

#endif // HAVE_OPENCL_STATIC

#endif // HAVE_OPENCL

#endif // __OPENCV_OCL_CL_RUNTIME_HPP__
