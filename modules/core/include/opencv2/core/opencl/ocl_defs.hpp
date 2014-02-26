// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

//#define CV_OPENCL_RUN_VERBOSE
//#define CV_OPENCL_RUN_ASSERT

#ifdef HAVE_OPENCL

#define CV_OPENCL_MESSAGE_PREFIX "\t\t\t\t"

#ifdef CV_OPENCL_RUN_VERBOSE
#define __CV_OPENCL_DUMP_RUNNING { printf(CV_OPENCL_MESSAGE_PREFIX "%s: OpenCL implementation is running\n", CV_Func); fflush(stdout); }
#define __CV_OPENCL_DUMP_PLAIN_RUNNING { printf(CV_OPENCL_MESSAGE_PREFIX "%s: Plain implementation is running\n", CV_Func); fflush(stdout); }
#else
#define __CV_OPENCL_DUMP_RUNNING
#define __CV_OPENCL_DUMP_PLAIN_RUNNING
#endif

#ifdef CV_OPENCL_RUN_ASSERT
#define __CV_OPENCL_RUN_AND_CHECK(func) bool _ocl_run_result = false; CV_Assert(_ocl_run_result = (func)); if (_ocl_run_result)
#else
#define __CV_OPENCL_RUN_AND_CHECK(func) if (func)
#endif

#define CV_OPENCL_RUN(condition, func, completion_operator) \
    { \
        if (cv::ocl::useOpenCL() && (condition)) \
        { \
            __CV_OPENCL_RUN_AND_CHECK(func) \
            { \
                 __CV_OPENCL_DUMP_RUNNING \
                 completion_operator \
            } \
        } \
        __CV_OPENCL_DUMP_PLAIN_RUNNING \
    }

#else // HAVE_OPENCL
#define CV_OPENCL_RUN(condition, func, completion_operator)
#endif

#define CV_OCL_RUN(condition, func) CV_OPENCL_RUN(condition, func, return;)
#define CV_OCL_RUN_(condition, func, ...) CV_OPENCL_RUN(condition, func, return __VA_ARGS__;)
