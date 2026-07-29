#ifndef OPENCV_DNN_METAL_OPS_CONVOLUTION_HPP
#define OPENCV_DNN_METAL_OPS_CONVOLUTION_HPP

#include "../../op_metal.hpp"

#ifdef HAVE_METAL
namespace cv { namespace dnn { namespace metal {

struct ConvolutionConfiguration
{
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
};

class ConvolutionOp final
{
public:
    static Ptr<BackendNode> create(const std::vector<Ptr<BackendWrapper>>& inputs,
                                   const std::vector<Ptr<BackendWrapper>>& outputs,
                                   const ConvolutionConfiguration& config);
};

}}}  // namespace cv::dnn::metal
#endif  // HAVE_METAL

#endif  // OPENCV_DNN_METAL_OPS_CONVOLUTION_HPP
