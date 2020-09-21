// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_WGPU_OP_PRIOR_BOX_HPP
#define OPENCV_DNN_WGPU_OP_PRIOR_BOX_HPP
#include "webgpu_common.hpp"
#include "webgpu_op_base.hpp"
namespace cv { namespace dnn { namespace webgpu {

#ifdef HAVE_WEBGPU

class OpPriorBox: public OpBase
{
public:
    OpPriorBox(float step_x,
               float step_y,
               bool clip,
               int num_priors,
               std::vector<float>& variance,
               std::vector<float>& offsets_x,
               std::vector<float>& offsets_y,
               std::vector<float>& box_widths,
               std::vector<float>& box_heights);
    ~OpPriorBox();
    bool forward(std::vector<Tensor>& in, Tensor& out);
    void reshapeOutTensor(std::vector<Tensor *>& in, Tensor& out);
    virtual bool forward(std::vector<Tensor>& ins,
                         std::vector<Tensor>& blobs,
                         std::vector<Tensor>& outs) CV_OVERRIDE;
private:
    bool computeGroupCount();

    int global_size_;
    int nthreads_;
    float step_x_;
    float step_y_;
    bool clip_;
    int num_priors_;
    std::vector<float> variance_;
    std::vector<float> offsets_x_;
    std::vector<float> offsets_y_;
    std::vector<float> box_widths_;
    std::vector<float> box_heights_;
    int img_h_;
    int img_w_;
    int in_h_;
    int in_w_;
    int out_channel_;
    int out_channel_size_;
    Tensor* tensor_offsets_x_= nullptr;
    Tensor* tensor_offsets_y_= nullptr;
    Tensor* tensor_widths_= nullptr;
    Tensor* tensor_heights_= nullptr;
    Tensor* tensor_variance_= nullptr;
};

#endif  // HAVE_WEBGPU

}}} // namespace cv::dnn::webgpu

#endif // OPENCV_DNN_WGPU_OP_PRIOR_BOX_HPP
