// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef OPENCV_DNN_ATTN_MASK_HPP
#define OPENCV_DNN_ATTN_MASK_HPP

#include "opencv2/core/hal/intrin.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cv { namespace dnn {

void apply_mask_int(Mat &att_weights, const Mat &att_mask,
        const int seq_len_kv, const int seq_len_q, const float min_val,
        const bool has_mask, const bool is_causal);

void apply_mask_float(Mat &att_weights, const Mat &att_mask,
        const int seq_len_kv, const int seq_len_q, const float min_val,
        const bool has_mask, const bool is_causal);

}}

#endif
