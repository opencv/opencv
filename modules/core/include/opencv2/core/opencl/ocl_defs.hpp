// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CORE_OPENCL_DEFS_HPP
#define OPENCV_CORE_OPENCL_DEFS_HPP

#include "opencv2/core/utility.hpp"
#include "cvconfig.h"
#include "opencv2/core/hal/backend_registry.hpp"
#include "opencv2/core/utils/logger.hpp"

namespace cv { namespace ocl {
#ifdef HAVE_OPENCL
/// Call is similar to useOpenCL() but doesn't try to load OpenCL runtime or create OpenCL context
CV_EXPORTS bool isOpenCLActivated();
#else
static inline bool isOpenCLActivated() { return false; }
#endif
}} // namespace


// CV_GPU_RUN(src, method, args...): dispatch to src's GPU backend, else fall through.
// e.g. CV_GPU_RUN(_src, resize, _src, _dst, dsize, sx, sy, interp)

#ifdef CV_GPU_RUN_VERBOSE

#define CV_GPU_RUN_(src, method, ...)                                   \
    {                                                                   \
        cv::hal::Backend* __gpu_b = nullptr;                            \
        if ((src).isUMat())                                             \
            __gpu_b = (src).getUMat().backend();                        \
        if (__gpu_b)                                                    \
        {                                                               \
            if (__gpu_b->method(__VA_ARGS__))                           \
            {                                                           \
                return;                                                 \
            }                                                           \
        }                                                               \
    }

#elif defined CV_GPU_RUN_ASSERT

#define CV_GPU_RUN_(src, method, ...)                                   \
    {                                                                   \
        cv::hal::Backend* __gpu_b = nullptr;                            \
        if ((src).isUMat())                                             \
            __gpu_b = (src).getUMat().backend();                        \
        if (__gpu_b)                                                    \
        {                                                               \
            if (__gpu_b->method(__VA_ARGS__))                           \
                return;                                                 \
            CV_Error(cv::Error::StsAssert,                              \
                "CV_GPU_RUN: GPU-backed source but op not handled");    \
        }                                                               \
    }

#else

// Normal mode — silent, dispatch only when source is GPU-backed
#define CV_GPU_RUN_(src, method, ...)                                   \
    {                                                                   \
        cv::hal::Backend* __gpu_b = nullptr;                            \
        if ((src).isUMat())                                             \
            __gpu_b = (src).getUMat().backend();                        \
        if (__gpu_b && __gpu_b->method(__VA_ARGS__))                    \
        {                                                               \
            return;                                                     \
        }                                                               \
    }

#endif

// Public macro — use in cv:: function bodies.
#define CV_GPU_RUN(src, method, ...)                                    \
    CV_GPU_RUN_(src, method, __VA_ARGS__)

//#define CV_OPENCL_RUN_ASSERT

#ifdef HAVE_OPENCL

#ifdef CV_OPENCL_RUN_VERBOSE
#define CV_OCL_RUN_(condition, func, ...)                                   \
    {                                                                       \
        if (cv::ocl::isOpenCLActivated() && (condition) && func)            \
        {                                                                   \
            printf("%s: OpenCL implementation is running\n", CV_Func);      \
            fflush(stdout);                                                 \
            CV_IMPL_ADD(CV_IMPL_OCL);                                       \
            return __VA_ARGS__;                                             \
        }                                                                   \
        else                                                                \
        {                                                                   \
            printf("%s: Plain implementation is running\n", CV_Func);       \
            fflush(stdout);                                                 \
        }                                                                   \
    }
#elif defined CV_OPENCL_RUN_ASSERT
#define CV_OCL_RUN_(condition, func, ...)                                   \
    {                                                                       \
        if (cv::ocl::isOpenCLActivated() && (condition))                    \
        {                                                                   \
            if(func)                                                        \
            {                                                               \
                CV_IMPL_ADD(CV_IMPL_OCL);                                   \
            }                                                               \
            else                                                            \
            {                                                               \
                CV_Error(cv::Error::StsAssert, #func);                      \
            }                                                               \
            return __VA_ARGS__;                                             \
        }                                                                   \
    }
#else
#define CV_OCL_RUN_(condition, func, ...)                                   \
try \
{ \
    if (cv::ocl::isOpenCLActivated() && (condition) && func)                \
    {                                                                       \
        CV_IMPL_ADD(CV_IMPL_OCL);                                           \
        return __VA_ARGS__;                                                 \
    } \
} \
catch (const cv::Exception& e) \
{ \
    CV_UNUSED(e); /* TODO: Add some logging here */ \
}
#endif

#else
#define CV_OCL_RUN_(condition, func, ...)
#endif

#define CV_OCL_RUN(condition, func) CV_OCL_RUN_(condition, func)

#endif // OPENCV_CORE_OPENCL_DEFS_HPP
