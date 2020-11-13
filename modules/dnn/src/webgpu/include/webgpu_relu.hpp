// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_WGPU_OP_RELU_HPP
#define OPENCV_DNN_WGPU_OP_RELU_HPP
#include "webgpu_common.hpp"
#include "webgpu_op_base.hpp"
namespace cv { namespace dnn { namespace webgpu {

#ifdef HAVE_WEBGPU

class OpReLU: public OpBase
{
public:
    OpReLU(const float slope = 1.f);
    bool forward(Tensor& in, Tensor& out);
    void reshapeOutTensor(Tensor& in, Tensor& out);
    virtual bool forward(std::vector<Tensor>& ins,
                         std::vector<Tensor>& blobs,
                         std::vector<Tensor>& outs) CV_OVERRIDE;
private:
    bool computeGroupCount();
    int total_;
    float slope_;
};

#endif  // HAVE_WEBGPU

}}} // namespace cv::dnn::webgpu

#endif // OPENCV_DNN_WGPU_OP_RELU_HPP
