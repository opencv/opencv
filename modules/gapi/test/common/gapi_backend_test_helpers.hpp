// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019 Intel Corporation

#ifndef OPENCV_GAPI_BACKEND_TEST_HELPERS_HPP
#define OPENCV_GAPI_BACKEND_TEST_HELPERS_HPP

#define __CORE_CPU [] () { return cv::compile_args(cv::gapi::core::cpu::kernels()); }
#define __CORE_FLUID [] () { return cv::compile_args(cv::gapi::core::fluid::kernels()); }
#define __CORE_GPU [] () { return cv::compile_args(cv::gapi::core::gpu::kernels()); }

#define REGISTER_FOR_ALL(RegisterF) \
    RegisterF(CPU, __CORE_CPU) \
    RegisterF(Fluid, __CORE_FLUID) \
    RegisterF(GPU, __CORE_GPU)

#endif //OPENCV_GAPI_BACKEND_TEST_HELPERS_HPP