#ifndef OPENCV_DNN_METAL_OPS_REDUCE_HPP
#define OPENCV_DNN_METAL_OPS_REDUCE_HPP

#include "../../op_metal.hpp"

#ifdef HAVE_METAL
namespace cv { namespace dnn { namespace metal {

enum class ReduceType
{
    MAX,
    MIN,
    MEAN,
    SUM,
    L1,
    L2,
    PROD,
    SUM_SQUARE,
    LOG_SUM,
    LOG_SUM_EXP
};

struct ReduceConfiguration
{
    ReduceType type = ReduceType::SUM;
    std::vector<int> axes;
    bool noopWithEmptyAxes = false;
};

class ReduceOp final
{
public:
    static Ptr<BackendNode> create(const std::vector<Ptr<BackendWrapper>>& inputs,
                                   const std::vector<Ptr<BackendWrapper>>& outputs,
                                   const ReduceConfiguration& config);
};

}}}  // namespace cv::dnn::metal
#endif  // HAVE_METAL

#endif  // OPENCV_DNN_METAL_OPS_REDUCE_HPP
