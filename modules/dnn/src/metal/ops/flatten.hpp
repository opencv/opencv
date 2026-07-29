#ifndef OPENCV_DNN_METAL_OPS_FLATTEN_HPP
#define OPENCV_DNN_METAL_OPS_FLATTEN_HPP

#include "../../op_metal.hpp"

#ifdef HAVE_METAL
namespace cv { namespace dnn { namespace metal {

class FlattenOp final
{
public:
    static Ptr<BackendNode> create(const std::vector<Ptr<BackendWrapper>>& inputs,
                                   const std::vector<Ptr<BackendWrapper>>& outputs);
};

}}}  // namespace cv::dnn::metal
#endif  // HAVE_METAL

#endif  // OPENCV_DNN_METAL_OPS_FLATTEN_HPP
