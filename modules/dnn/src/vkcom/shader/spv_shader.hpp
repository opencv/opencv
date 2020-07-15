// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_SPV_SHADER_HPP
#define OPENCV_DNN_SPV_SHADER_HPP


namespace cv { namespace dnn { namespace vkcom {

extern const unsigned int dw_conv_spv[1760];
extern const unsigned int permute_spv[765];
extern const unsigned int conv48_spv[7458];
extern const unsigned int lrn_spv[1845];
extern const unsigned int concat_spv[541];
extern const unsigned int avg_pool_spv[1538];
extern const unsigned int softmax_spv[1496];
extern const unsigned int prior_box_spv[1480];
extern const unsigned int max_pool_spv[1449];
extern const unsigned int relu_spv[502];
extern const unsigned int conv_spv[1894];

}}} // namespace cv::dnn::vkcom

#endif /* OPENCV_DNN_SPV_SHADER_HPP */
