#ifndef OPENCV_DNN_METAL_OPS_BATCH_NORM_HPP
#define OPENCV_DNN_METAL_OPS_BATCH_NORM_HPP

#include "../../op_metal.hpp"

#ifdef HAVE_METAL
namespace cv { namespace dnn { namespace metal {

struct BatchNormConfiguration
{
    Mat scale;
    Mat bias;
    int axis = 1;
};

class BatchNormOp final
{
public:
    static Ptr<BackendNode> create(const std::vector<Ptr<BackendWrapper>>& inputs,
                                   const std::vector<Ptr<BackendWrapper>>& outputs,
                                   const BatchNormConfiguration& config);
};

}}}  // namespace cv::dnn::metal
#endif  // HAVE_METAL

#endif  // OPENCV_DNN_METAL_OPS_BATCH_NORM_HPP
