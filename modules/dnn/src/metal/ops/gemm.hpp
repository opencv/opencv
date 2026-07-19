// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_METAL_OPS_GEMM_HPP
#define OPENCV_DNN_METAL_OPS_GEMM_HPP

#include "../../op_metal.hpp"

#ifdef HAVE_METAL
namespace cv { namespace dnn { namespace metal {

struct GemmConfiguration
{
    Mat weights;
    Mat bias;
    bool hasBias = false;
    bool transA = false;
    bool transB = false;
    bool flattenA = true;
    float alpha = 1.0f;
    float beta = 1.0f;
    int realBiasRank = -1;
};

class GemmOp final
{
public:
    static Ptr<BackendNode> create(const std::vector<Ptr<BackendWrapper>>& inputs,
                                   const std::vector<Ptr<BackendWrapper>>& outputs,
                                   const GemmConfiguration& config);
};

}}}  // namespace cv::dnn::metal
#endif  // HAVE_METAL

#endif  // OPENCV_DNN_METAL_OPS_GEMM_HPP
