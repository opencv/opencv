// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DNN_INT8LAYERS_INT8_VNNI_HPP__
#define __OPENCV_DNN_INT8LAYERS_INT8_VNNI_HPP__

#include "../layers/conv2_common.hpp"

namespace cv
{
namespace dnn
{

// Entry points of the AVX-VNNI int8 kernels. The kernels live in
// layers_common_vnni.simd.hpp and conv2_int8_vnni_kernels.simd.hpp and are built
// only when CMake found a toolchain that can encode AVX-VNNI (optimization
// AVX_VNNI). Ask CV_CPU_HAS_SUPPORT_AVX_VNNI before calling any of them: it is a
// compile-time 0 when the path was not built and a runtime CPU check otherwise.
// Calling one anyway raises an error instead of executing an illegal instruction.

void fastConvVNNI( const int8_t* weights, size_t wstep, const int* bias,
                   const uint8_t* rowbuf, int* output, const int* outShape,
                   int blockSize, int vecsize, int vecsize_aligned, int outZp,
                   const float* multiplier, bool initOutput, bool finalOutput );

void fastGEMM1TVNNI( const uint8_t* vec, const int8_t* weights,
                     size_t wstep, const int* bias, const float* multiplier,
                     int* dst, int nvecs, int vecsize, int outZp );

void convInt8BlockVNNI(const void* inp_, const void* residual_,
                       void* out_, const ConvState& cs,
                       const void* weightsVNNI_,
                       const int* biasVNNI, const float* multiplier,
                       int inp_zp, int out_zp,
                       const int8_t* activLUT,
                       bool inputIsU8);

}
}

#endif
