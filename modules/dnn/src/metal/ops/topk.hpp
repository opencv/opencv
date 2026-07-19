#ifndef OPENCV_DNN_METAL_OPS_TOPK_HPP
#define OPENCV_DNN_METAL_OPS_TOPK_HPP

#include "../../op_metal.hpp"

#ifdef HAVE_METAL
namespace cv { namespace dnn { namespace metal {

struct TopKConfiguration
{
    int axis = -1;
    int k = 1;
    bool largest = true;
};

class TopKOp final
{
public:
    static Ptr<BackendNode> create(const std::vector<Ptr<BackendWrapper>>& inputs,
                                   const std::vector<Ptr<BackendWrapper>>& outputs,
                                   const TopKConfiguration& config);
};

}}}  // namespace cv::dnn::metal
#endif  // HAVE_METAL

#endif  // OPENCV_DNN_METAL_OPS_TOPK_HPP
