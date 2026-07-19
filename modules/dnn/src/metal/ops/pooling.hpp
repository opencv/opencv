#ifndef OPENCV_DNN_METAL_OPS_POOLING_HPP
#define OPENCV_DNN_METAL_OPS_POOLING_HPP

#include "../../op_metal.hpp"

#ifdef HAVE_METAL
namespace cv { namespace dnn { namespace metal {

struct AvgPool2DConfiguration
{
    size_t kernelHeight = 0;
    size_t kernelWidth = 0;
    size_t strideHeight = 1;
    size_t strideWidth = 1;
    size_t padTop = 0;
    size_t padBottom = 0;
    size_t padLeft = 0;
    size_t padRight = 0;
    bool includePadding = false;
};

class AvgPool2DOp final
{
public:
    static Ptr<BackendNode> create(const std::vector<Ptr<BackendWrapper>>& inputs,
                                   const std::vector<Ptr<BackendWrapper>>& outputs,
                                   const AvgPool2DConfiguration& config);
};

struct MaxPoolingConfiguration
{
    size_t kernelHeight = 0;
    size_t kernelWidth = 0;
    size_t strideHeight = 1;
    size_t strideWidth = 1;
    size_t padTop = 0;
    size_t padLeft = 0;
};

class MaxPoolingOp final
{
public:
    static Ptr<BackendNode> create(const std::vector<Ptr<BackendWrapper>>& inputs,
                                   const std::vector<Ptr<BackendWrapper>>& outputs,
                                   const MaxPoolingConfiguration& config);
};

}}}  // namespace cv::dnn::metal
#endif  // HAVE_METAL

#endif  // OPENCV_DNN_METAL_OPS_POOLING_HPP
