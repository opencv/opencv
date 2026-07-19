// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_METAL_OPS_DEPTH_SPACE_OPS_HPP
#define OPENCV_DNN_METAL_OPS_DEPTH_SPACE_OPS_HPP

#include "../../op_metal.hpp"

#ifdef HAVE_METAL
namespace cv { namespace dnn { namespace metal {

enum class DepthSpaceOperation
{
    DEPTH_TO_SPACE_DCR,
    DEPTH_TO_SPACE_CRD,
    SPACE_TO_DEPTH
};

class DepthSpaceOpsOp final
{
public:
    static Ptr<BackendNode> create(const std::vector<Ptr<BackendWrapper>>& inputs,
                                   const std::vector<Ptr<BackendWrapper>>& outputs,
                                   DepthSpaceOperation operation,
                                   int blockSize);
};

}}}  // namespace cv::dnn::metal
#endif  // HAVE_METAL

#endif  // OPENCV_DNN_METAL_OPS_DEPTH_SPACE_OPS_HPP
