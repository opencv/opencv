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

namespace cv { namespace ocl {
#ifdef HAVE_OPENCL
/// Call is similar to useOpenCL() but doesn't try to load OpenCL runtime or create OpenCL context
CV_EXPORTS bool isOpenCLActivated();
#else
static inline bool isOpenCLActivated() { return false; }
#endif
}} // namespace


// ─────────────────────────────────────────────────────────────────────────────
// CV_GPU_RUN — dispatches to the first registered GPU backend that supports
// op_id.  Mirrors the structure of CV_OCL_RUN_ below.
// Must appear before CV_OCL_RUN in this file.
// ─────────────────────────────────────────────────────────────────────────────

#ifdef CV_GPU_RUN_VERBOSE
#define CV_GPU_RUN_(op_id, src, dst, ...)                                       \
    {                                                                           \
        cv::hal::Backend* _cv_gpu_b = cv::hal::findBackend(op_id);              \
        if (_cv_gpu_b && _cv_gpu_b->run((op_id), (src), (dst)))                 \
        {                                                                       \
            printf("%s: GPU backend is running\n", CV_Func);                    \
            fflush(stdout);                                                     \
            return __VA_ARGS__;                                                 \
        }                                                                       \
        else                                                                    \
        {                                                                       \
            printf("%s: CPU implementation is running\n", CV_Func);             \
            fflush(stdout);                                                     \
        }                                                                       \
    }
#elif defined CV_GPU_RUN_ASSERT
#define CV_GPU_RUN_(op_id, src, dst, ...)                                       \
    {                                                                           \
        cv::hal::Backend* _cv_gpu_b = cv::hal::findBackend(op_id);              \
        if (_cv_gpu_b)                                                          \
        {                                                                       \
            if (_cv_gpu_b->run((op_id), (src), (dst)))                          \
            {                                                                   \
                return __VA_ARGS__;                                             \
            }                                                                   \
            else                                                                \
            {                                                                   \
                CV_Error(cv::Error::StsAssert,                                  \
                         "GPU backend run() returned false for this op_id");    \
            }                                                                   \
        }                                                                       \
    }
#else
#define CV_GPU_RUN_(op_id, src, dst, ...)                                       \
    {                                                                           \
        cv::hal::Backend* _cv_gpu_b = cv::hal::findBackend(op_id);              \
        if (_cv_gpu_b && _cv_gpu_b->run((op_id), (src), (dst)))                 \
        {                                                                       \
            return __VA_ARGS__;                                                 \
        }                                                                       \
    }
#endif

#define CV_GPU_RUN(op_id, src, dst) CV_GPU_RUN_(op_id, src, dst)

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
