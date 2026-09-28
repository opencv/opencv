// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DNN_INT8LAYERS_CONV2_INT8_KERNELS_HPP__
#define __OPENCV_DNN_INT8LAYERS_CONV2_INT8_KERNELS_HPP__

#include "../layers/conv2_common.hpp"

namespace cv
{
namespace dnn
{

// Entry points of the int8 convolution kernels. The kernels themselves live in
// conv2_int8_kernels.simd.hpp and are dispatched in conv2_int8_kernels.dispatch.cpp.

void convInt8Block(const void* inp_, const void* residual_,
                   void* out_, const ConvState& cs,
                   const void* weights_,
                   const void* weightsVNNI_,
                   const int* bias, const int* biasVNNI_,
                   const float* multiplier,
                   int inp_zp, int out_zp,
                   const int8_t* activLUT,
                   bool inputIsU8);

// AVX-VNNI kernel of the legacy int8 convolution layer. Built only when CMake found
// a toolchain that can encode AVX-VNNI, so ask CV_CPU_HAS_SUPPORT_AVX_VNNI before
// calling it: it is a compile-time 0 when the path was not built and a runtime CPU
// check otherwise. Calling it anyway raises an error instead of executing an
// illegal instruction.
void fastConvVNNI( const int8_t* weights, size_t wstep, const int* bias,
                   const uint8_t* rowbuf, int* output, const int* outShape,
                   int blockSize, int vecsize, int vecsize_aligned, int outZp,
                   const float* multiplier, bool initOutput, bool finalOutput );

}
}

#endif
