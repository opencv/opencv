// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_METAL_OPS_PADDING_HPP
#define OPENCV_DNN_METAL_OPS_PADDING_HPP

#include "../../op_metal.hpp"

#ifdef HAVE_METAL
namespace cv { namespace dnn { namespace metal {

enum class PaddingType
{
    CONSTANT,
    REFLECT,
    EDGE
};

struct PaddingConfiguration
{
    PaddingType type = PaddingType::CONSTANT;
    std::vector<std::pair<int, int>> paddings;
    double value = 0.0;
};

class PaddingOp final
{
public:
    static Ptr<BackendNode> create(const std::vector<Ptr<BackendWrapper>>& inputs,
                                   const std::vector<Ptr<BackendWrapper>>& outputs,
                                   const PaddingConfiguration& config);
};

}}}  // namespace cv::dnn::metal
#endif  // HAVE_METAL

#endif  // OPENCV_DNN_METAL_OPS_PADDING_HPP
