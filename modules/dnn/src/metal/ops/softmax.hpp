#ifndef OPENCV_DNN_METAL_OPS_SOFTMAX_HPP
#define OPENCV_DNN_METAL_OPS_SOFTMAX_HPP

#include "../../op_metal.hpp"

#ifdef HAVE_METAL
namespace cv { namespace dnn { namespace metal {

struct SoftmaxConfiguration
{
    int axis = 0;
    bool logSoftmax = false;
    float scale = 1.0f;
};

class SoftmaxOp final
{
public:
    static Ptr<BackendNode> create(const std::vector<Ptr<BackendWrapper>>& inputs,
                                   const std::vector<Ptr<BackendWrapper>>& outputs,
                                   const SoftmaxConfiguration& config);
};

}}}  // namespace cv::dnn::metal
#endif  // HAVE_METAL

#endif  // OPENCV_DNN_METAL_OPS_SOFTMAX_HPP
