#ifndef OPENCV_DNN_METAL_OPS_LAYER_NORM_HPP
#define OPENCV_DNN_METAL_OPS_LAYER_NORM_HPP

#include "../../op_metal.hpp"

#ifdef HAVE_METAL
namespace cv { namespace dnn { namespace metal {

struct LayerNormConfiguration
{
    Mat scale;
    Mat bias;
    int axis = -1;
    float epsilon = 1e-5f;
    bool hasBias = false;
};

class LayerNormOp final
{
public:
    static Ptr<BackendNode> create(const std::vector<Ptr<BackendWrapper>>& inputs,
                                   const std::vector<Ptr<BackendWrapper>>& outputs,
                                   const LayerNormConfiguration& config);
};

}}}  // namespace cv::dnn::metal
#endif  // HAVE_METAL

#endif  // OPENCV_DNN_METAL_OPS_LAYER_NORM_HPP
