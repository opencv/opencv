#ifndef OPENCV_DNN_METAL_OPS_OPERATION_HPP
#define OPENCV_DNN_METAL_OPS_OPERATION_HPP

namespace cv { namespace dnn { namespace metal {

class Context;

class Operation
{
public:
    virtual ~Operation() = default;
    virtual void forward(Context& context) = 0;
};

}}}  // namespace cv::dnn::metal

#endif  // OPENCV_DNN_METAL_OPS_OPERATION_HPP
