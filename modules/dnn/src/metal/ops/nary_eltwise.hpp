#ifndef OPENCV_DNN_METAL_OPS_NARY_ELTWISE_HPP
#define OPENCV_DNN_METAL_OPS_NARY_ELTWISE_HPP

#include "../../op_metal.hpp"

#ifdef HAVE_METAL
namespace cv { namespace dnn { namespace metal {

enum class NaryOperation
{
    Add = 0,
    Multiply = 1,
    Subtract = 2,
    Divide = 3,
    Maximum = 4,
    Minimum = 5
};

class NaryEltwiseOp final
{
public:
    static Ptr<BackendNode> create(const std::vector<Ptr<BackendWrapper>>& inputs,
                                   const std::vector<Ptr<BackendWrapper>>& outputs,
                                   NaryOperation operation);
};

}}}  // namespace cv::dnn::metal
#endif  // HAVE_METAL

#endif  // OPENCV_DNN_METAL_OPS_NARY_ELTWISE_HPP
