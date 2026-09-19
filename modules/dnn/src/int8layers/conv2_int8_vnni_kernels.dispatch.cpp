// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "int8_vnni.hpp"

#include "conv2_int8_vnni_kernels.simd.hpp"
#include "int8layers/conv2_int8_vnni_kernels.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX_VNNI,BASELINE based on CMakeLists.txt content

namespace cv
{
namespace dnn
{

void convInt8BlockVNNI(const void* inp_, const void* residual_,
                       void* out_, const ConvState& cs,
                       const void* weightsVNNI_,
                       const int* biasVNNI, const float* multiplier,
                       int inp_zp, int out_zp,
                       const int8_t* activLUT,
                       bool inputIsU8)
{
    CV_CPU_CALL_AVX_VNNI(convInt8BlockVNNI, (inp_, residual_, out_, cs, weightsVNNI_,
                                             biasVNNI, multiplier, inp_zp, out_zp,
                                             activLUT, inputIsU8));
    CV_Error(Error::StsNotImplemented, "DNN/INT8: AVX-VNNI convolution kernel is not available");
}

}
}
