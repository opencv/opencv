#ifndef OPENCV_DNN_METAL_OPS_SCALE_HPP
#define OPENCV_DNN_METAL_OPS_SCALE_HPP

#include "../../op_metal.hpp"

#ifdef HAVE_METAL
namespace cv { namespace dnn { namespace metal {

struct AffineConfiguration
{
    enum class DynamicParameter
    {
        NONE,
        SCALE,
        BIAS
    };

    Mat scale;
    Mat bias;
    int axis = 0;
    DynamicParameter dynamicParameter = DynamicParameter::NONE;
};

class AffineOp final
{
public:
    static Ptr<BackendNode> create(const std::vector<Ptr<BackendWrapper>>& inputs,
                                   const std::vector<Ptr<BackendWrapper>>& outputs,
                                   const AffineConfiguration& config);
};

}}}  // namespace cv::dnn::metal
#endif  // HAVE_METAL

#endif  // OPENCV_DNN_METAL_OPS_SCALE_HPP
