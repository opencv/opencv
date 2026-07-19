#ifndef OPENCV_DNN_METAL_OPS_PERMUTE_HPP
#define OPENCV_DNN_METAL_OPS_PERMUTE_HPP

#include "../../op_metal.hpp"

#ifdef HAVE_METAL
namespace cv { namespace dnn { namespace metal {

class PermuteOp final
{
public:
    static Ptr<BackendNode> create(const std::vector<Ptr<BackendWrapper>>& inputs,
                                   const std::vector<Ptr<BackendWrapper>>& outputs,
                                   const std::vector<size_t>& order);
};

}}}  // namespace cv::dnn::metal
#endif  // HAVE_METAL

#endif  // OPENCV_DNN_METAL_OPS_PERMUTE_HPP
