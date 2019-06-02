// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// this file is a stub and will be removed once actual code is added

#include "../precomp.hpp"

#include <cuda_runtime.h>

#ifndef HAVE_CUDA
#   error "CUDA files should not be compiled if CUDA was not enabled"
#endif

__global__ void cuda4dnn_build_test_kernel(float* addr) {
    int idx = threadIdx.x;
    addr[idx] = 0.0;
}
