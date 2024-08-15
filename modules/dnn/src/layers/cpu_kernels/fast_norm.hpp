// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_FAST_NORM_HPP
#define OPENCV_DNN_FAST_NORM_HPP

#include <opencv2/dnn/shape_utils.hpp>

namespace cv { namespace dnn {

// Normalization speedup by multi-threading, mainly for Caffe MVN layer which has normalize_variance parameter.
void fastNorm(const Mat &input, Mat &output, float epsilon, size_t normalized_axis = 0, bool normalize_variance = true);

// Normalization speedup by multi-threading with absent bias. Mainly for LayerNormalization.
void fastNorm(const Mat &input, const Mat &scale, Mat &output, float epsilon, size_t normalized_axis = 0);

// Normalization speedup by multi-threading with scale and bias. Mainly for LayerNormalization.
void fastNorm(const Mat &input, const Mat &scale, const Mat &bias, Mat &output, float epsilon, size_t normalized_axis = 0);

// Channel-wise Normalization speedup by multi-threading. Scale and bias should have the same shape (C). Input should have dimension >= 3.
void fastNormChannel(const Mat &input, const Mat &scale, const Mat &bias, Mat &output, float epsilon);

// Group-wise Normalization speedup by multi-threading. Scale and bias should have the same shape (C). Input should have dimension >= 3.
void fastNormGroup(const Mat &input, const Mat &scale, const Mat &bias, Mat &output, float epsilon, size_t num_groups);

}} // cv::dnn

#endif // OPENCV_DNN_FAST_NORM_HPP
