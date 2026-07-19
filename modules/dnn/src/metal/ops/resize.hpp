#ifndef OPENCV_DNN_METAL_OPS_RESIZE_HPP
#define OPENCV_DNN_METAL_OPS_RESIZE_HPP

#include "../../op_metal.hpp"

#ifdef HAVE_METAL
namespace cv { namespace dnn { namespace metal {

enum class InterpolationType
{
    Nearest = 0,
    Bilinear
};

struct ResizeConfiguration
{
    InterpolationType interpolation = InterpolationType::Nearest;
    bool alignCorners = false;
    bool halfPixelCenters = false;
};

class ResizeOp final
{
public:
    static Ptr<BackendNode> create(const std::vector<Ptr<BackendWrapper>>& inputs,
                                   const std::vector<Ptr<BackendWrapper>>& outputs,
                                   const ResizeConfiguration& config);
};

}}}  // namespace cv::dnn::metal
#endif  // HAVE_METAL

#endif  // OPENCV_DNN_METAL_OPS_RESIZE_HPP
