// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include "opencv2/core/base.hpp"

__host__ inline void hipCheckError_(hipError_t err, const char* file, int line, const char* func) {
    if (err != hipSuccess) {
        (void)hipGetLastError();
        cv::error(cv::Error::GpuApiCallError, hipGetErrorString(err), func, file, line);
    }
}
#define CV_HIP_SAFE_CALL(expr) ::hipCheckError_((expr), __FILE__, __LINE__, CV_Func)
