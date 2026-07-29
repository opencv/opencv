#ifndef OPENCV_DNN_METAL_OPS_FULLY_CONNECTED_HPP
#define OPENCV_DNN_METAL_OPS_FULLY_CONNECTED_HPP

#include "../../op_metal.hpp"

#ifdef HAVE_METAL
namespace cv { namespace dnn { namespace metal {

struct FullyConnectedConfiguration
{
    Mat weights;
    Mat bias;
    int axis = 0;
};

class FullyConnectedOp final
{
public:
    static Ptr<BackendNode> create(const std::vector<Ptr<BackendWrapper>>& inputs,
                                   const std::vector<Ptr<BackendWrapper>>& outputs,
                                   const FullyConnectedConfiguration& config);
};

}}}  // namespace cv::dnn::metal
#endif  // HAVE_METAL

#endif  // OPENCV_DNN_METAL_OPS_FULLY_CONNECTED_HPP
