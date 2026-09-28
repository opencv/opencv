// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "conv2_int8_kernels.hpp"

#include "conv2_int8_kernels.simd.hpp"
#include "int8layers/conv2_int8_kernels.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX_VNNI,RVV,AVX2,BASELINE based on CMakeLists.txt content

namespace cv
{
namespace dnn
{

void convInt8Block(const void* inp_, const void* residual_,
                   void* out_, const ConvState& cs,
                   const void* weights_,
                   const void* weightsVNNI_,
                   const int* bias, const int* biasVNNI_,
                   const float* multiplier,
                   int inp_zp, int out_zp,
                   const int8_t* activLUT,
                   bool inputIsU8)
{
    CV_CPU_DISPATCH(convInt8Block, (inp_, residual_, out_, cs, weights_, weightsVNNI_,
                                    bias, biasVNNI_, multiplier, inp_zp, out_zp,
                                    activLUT, inputIsU8),
                    CV_CPU_DISPATCH_MODES_ALL);
}

void fastConvVNNI( const int8_t* weights, size_t wstep, const int* bias,
                   const uint8_t* rowbuf, int* output, const int* outShape,
                   int blockSize, int vecsize, int vecsize_aligned, int outZp,
                   const float* multiplier, bool initOutput, bool finalOutput )
{
    CV_CPU_CALL_AVX_VNNI(fastConvVNNI, (weights, wstep, bias, rowbuf, output, outShape,
                                        blockSize, vecsize, vecsize_aligned, outZp,
                                        multiplier, initOutput, finalOutput));
    CV_Error(Error::StsNotImplemented, "DNN/INT8: AVX-VNNI convolution kernel is not available");
}

}
}
