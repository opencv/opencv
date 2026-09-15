// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "int8_vnni.hpp"

#include "layers_common_vnni.simd.hpp"
#include "int8layers/layers_common_vnni.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX_VNNI,BASELINE based on CMakeLists.txt content

namespace cv
{
namespace dnn
{

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

void fastGEMM1TVNNI( const uint8_t* vec, const int8_t* weights,
                     size_t wstep, const int* bias, const float* multiplier,
                     int* dst, int nvecs, int vecsize, int outZp )
{
    CV_CPU_CALL_AVX_VNNI(fastGEMM1TVNNI, (vec, weights, wstep, bias, multiplier,
                                          dst, nvecs, vecsize, outZp));
    CV_Error(Error::StsNotImplemented, "DNN/INT8: AVX-VNNI GEMM kernel is not available");
}

}
}
