#ifndef __OPENCV_OCL_CL_RUNTIME_HPP__
#define __OPENCV_OCL_CL_RUNTIME_HPP__

#ifdef HAVE_OPENCL

#if defined(HAVE_OPENCL12)
#include "cl_runtime_opencl12.hpp"
#elif defined(HAVE_OPENCL11)
#include "cl_runtime_opencl11.hpp"
#else
#error Invalid OpenCL configuration
#endif

#endif

#endif // __OPENCV_OCL_CL_RUNTIME_HPP__
