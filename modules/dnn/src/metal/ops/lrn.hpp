// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_METAL_OPS_LRN_HPP
#define OPENCV_DNN_METAL_OPS_LRN_HPP

#include "../../op_metal.hpp"

#ifdef HAVE_METAL
namespace cv { namespace dnn { namespace metal {

enum class LRNType
{
    ACROSS_CHANNELS,
    WITHIN_CHANNEL
};

struct LRNConfiguration
{
    LRNType type = LRNType::ACROSS_CHANNELS;
    int localSize = 5;
    float alpha = 0.0001f;
    float beta = 0.75f;
    float bias = 1.0f;
    bool normBySize = true;
};

class LRNOp final
{
public:
    static Ptr<BackendNode> create(const std::vector<Ptr<BackendWrapper>>& inputs,
                                   const std::vector<Ptr<BackendWrapper>>& outputs,
                                   const LRNConfiguration& config);
};

}}}  // namespace cv::dnn::metal
#endif  // HAVE_METAL

#endif  // OPENCV_DNN_METAL_OPS_LRN_HPP
