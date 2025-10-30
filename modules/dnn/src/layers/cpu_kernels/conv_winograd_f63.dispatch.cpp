// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "convolution.hpp"
#include "conv_winograd_f63.simd.hpp"
#include "layers/cpu_kernels/conv_winograd_f63.simd_declarations.hpp"

namespace cv {
namespace dnn {

cv::dnn::Winofunc getWinofunc_F32()
{
    CV_CPU_DISPATCH(getWinofunc_F32, (), CV_CPU_DISPATCH_MODES_ALL);
}

cv::dnn::Winofunc getWinofunc_F16()
{
    CV_CPU_DISPATCH(getWinofunc_F16, (), CV_CPU_DISPATCH_MODES_ALL);
}

}} // namespace cv::dnn::
