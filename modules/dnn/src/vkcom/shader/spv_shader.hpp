// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SPV_SHADER_HPP
#define OPENCV_DNN_SPV_SHADER_HPP


namespace cv { namespace dnn { namespace vkcom {

extern const unsigned int conv_1x1_fast_spv[3134];
extern const unsigned int conv_depthwise_spv[2092];
extern const unsigned int conv_depthwise_3x3_spv[1977];
extern const unsigned int conv_implicit_gemm_spv[3565];
extern const unsigned int gemm_spv[2902];
extern const unsigned int nary_eltwise_binary_forward_spv[1757];

extern std::map<std::string, std::pair<const unsigned int *, size_t> > SPVMaps;

void initSPVMaps();

}}} // namespace cv::dnn::vkcom

#endif /* OPENCV_DNN_SPV_SHADER_HPP */
