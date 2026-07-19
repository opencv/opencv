// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_METAL_OPS_DECONVOLUTION_HPP
#define OPENCV_DNN_METAL_OPS_DECONVOLUTION_HPP

#include "../../op_metal.hpp"

#ifdef HAVE_METAL
namespace cv { namespace dnn { namespace metal {

struct DeconvolutionConfiguration
{
    enum class PaddingMode
    {
        MANUAL,
        VALID,
        SAME
    };

    Mat weights;
    Mat bias;
    size_t groups = 1;
    size_t kernelHeight = 0;
    size_t kernelWidth = 0;
    size_t strideHeight = 1;
    size_t strideWidth = 1;
    size_t dilationHeight = 1;
    size_t dilationWidth = 1;
    size_t padTop = 0;
    size_t padLeft = 0;
    PaddingMode paddingMode = PaddingMode::MANUAL;
};

class DeconvolutionOp final
{
public:
    static Ptr<BackendNode> create(const std::vector<Ptr<BackendWrapper>>& inputs,
                                   const std::vector<Ptr<BackendWrapper>>& outputs,
                                   const DeconvolutionConfiguration& config);
};

}}}  // namespace cv::dnn::metal
#endif  // HAVE_METAL

#endif  // OPENCV_DNN_METAL_OPS_DECONVOLUTION_HPP
