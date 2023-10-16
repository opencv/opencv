// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_FAST_NORM_HPP
#define OPENCV_DNN_FAST_NORM_HPP

#include <opencv2/dnn/shape_utils.hpp>

namespace cv { namespace dnn {

void fastNorm(const Mat &input, const Mat &scale, const Mat &bias, Mat &output, float epsilon, int axis = 0);

}} // cv::dnn

#endif // OPENCV_DNN_FAST_NORM_HPP
