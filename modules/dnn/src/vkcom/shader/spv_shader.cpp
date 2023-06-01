// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "spv_shader.hpp"

namespace cv { namespace dnn { namespace vkcom {

std::map<std::string, std::pair<const unsigned int *, size_t> > SPVMaps;

void initSPVMaps()
{
    SPVMaps.insert(std::make_pair("conv_1x1_fast_spv", std::make_pair(conv_1x1_fast_spv, 3134)));
    SPVMaps.insert(std::make_pair("gemm_spv", std::make_pair(gemm_spv, 2902)));
    SPVMaps.insert(std::make_pair("conv_depthwise_3x3_spv", std::make_pair(conv_depthwise_3x3_spv, 1977)));
    SPVMaps.insert(std::make_pair("conv_implicit_gemm_spv", std::make_pair(conv_implicit_gemm_spv, 3565)));
    SPVMaps.insert(std::make_pair("conv_depthwise_spv", std::make_pair(conv_depthwise_spv, 2092)));
}

}}} // namespace cv::dnn::vkcom
