// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef OPENCV_DNN_FAST_ATTN
#define OPENCV_DNN_FAST_ATTN

#include "opencv2/core/hal/intrin.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cv { namespace dnn {

void fused_softmax_softcap_mask(
        Mat &att_weights,const Mat &att_mask,
        const float softcap, const bool do_softcap,
        const float threshold,
        const float min_val, const bool has_mask, const bool is_causal
);

}}

#endif
