// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// this file is a stub and will be removed once actual code is added

#include "../precomp.hpp"

#ifndef HAVE_CUDA
#   error "CUDA4DNN should be enabled iff CUDA and cuDNN were found"
#endif

#include <cudnn.h>

void cuda4dnn_build_test_func() {
    auto ver = cudnnGetVersion();
    CV_UNUSED(ver);
}
