// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_FAST_CONVOLUTION_INTELNALS_HPP
#define OPENCV_FAST_CONVOLUTION_INTELNALS_HPP

#include "fast_convolution.hpp"
#include "opencv2/core/hal/hal.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "../layers_common.hpp"

namespace cv { namespace dnn {

void runDepthwise(InputArray _input, OutputArray _output, const Ptr<FastConv2d>& conv, float minval, float maxval,
        ActivationLayer* activ, bool ifMinMaxAct);

// winograd init
void initWinograd63(Ptr<FastConv2d>& conv, float* src_weight, int K, int C);

int runWinograd63(InputArray _input, OutputArray _output, const Ptr<FastConv2d>& conv, int ntasks,
                  float minval, float maxval, ActivationLayer* activ, bool ifMinMaxAct);

}}  // namespace cv::dnn
#endif //OPENCV_FAST_CONVOLUTION_INTELNALS_HPP
