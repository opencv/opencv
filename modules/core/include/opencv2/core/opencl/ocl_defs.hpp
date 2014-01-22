// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#ifdef HAVE_OPENCL

#ifdef CV_OPENCL_RUN_VERBOSE
#define CV_OCL_RUN(condition, func)                                         \
    {                                                                       \
        if (cv::ocl::useOpenCL() && (condition) && func)                    \
        {                                                                   \
            printf("%s: OpenCL implementation is running\n", CV_Func);      \
            fflush(stdout);                                                 \
            return;                                                         \
        }                                                                   \
        else                                                                \
        {                                                                   \
            printf("%s: Plain implementation is running\n", CV_Func);       \
            fflush(stdout);                                                 \
        }                                                                   \
    }
#else
#define CV_OCL_RUN(condition, func)                                         \
    if (cv::ocl::useOpenCL() && (condition) && func)                        \
        return;
#endif

#else
#define CV_OCL_RUN(condition, func)
#endif

