// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_SPV_SHADER_HPP
#define OPENCV_DNN_SPV_SHADER_HPP


namespace cv { namespace dnn { namespace vkcom {

extern const unsigned int dw_conv_spv[1659];
extern const unsigned int permute_spv[765];
extern const unsigned int lrn_spv[1845];
extern const unsigned int concat_spv[541];
extern const unsigned int avg_pool_spv[1542];
extern const unsigned int softmax_spv[1440];
extern const unsigned int prior_box_spv[1484];
extern const unsigned int max_pool_spv[1453];
extern const unsigned int relu_spv[502];
extern const unsigned int conv_spv[1863];

}}} // namespace cv::dnn::vkcom

#endif /* OPENCV_DNN_SPV_SHADER_HPP */
