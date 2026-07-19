// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_METAL_OPS_SLICE_HPP
#define OPENCV_DNN_METAL_OPS_SLICE_HPP

#include "../../op_metal.hpp"

#ifdef HAVE_METAL
namespace cv { namespace dnn { namespace metal {

struct SliceConfiguration
{
    std::vector<std::vector<Range>> ranges;
    std::vector<std::vector<int>> steps;
    std::vector<int> flippedDimensions;
};

class SliceOp final
{
public:
    static Ptr<BackendNode> create(const std::vector<Ptr<BackendWrapper>>& inputs,
                                   const std::vector<Ptr<BackendWrapper>>& outputs,
                                   const SliceConfiguration& config);
};

}}}  // namespace cv::dnn::metal
#endif  // HAVE_METAL

#endif  // OPENCV_DNN_METAL_OPS_SLICE_HPP
