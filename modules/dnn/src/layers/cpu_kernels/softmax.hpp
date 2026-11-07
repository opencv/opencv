// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// This file is modified from the ficus (https://github.com/vpisarev/ficus/blob/master/lib/NN/OpNN.fx).
// Here is the original license:
/*
    This file is a part of ficus language project.
    See ficus/LICENSE for the licensing terms
*/

#ifndef OPENCV_DNN_SOFTMAX_HPP
#define OPENCV_DNN_SOFTMAX_HPP

#include "opencv2/core/hal/intrin.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cv { namespace dnn {

void softmax(Mat &dst, const Mat &src, int axis, int axisBias, int axisStep);

void softmax(Mat &dst, const Mat &src, int axis);

void logSoftmax(Mat &dst, const Mat &src, int axis);

}} // cv::dnn

#endif // OPENCV_DNN_SOFTMAX_HPP
