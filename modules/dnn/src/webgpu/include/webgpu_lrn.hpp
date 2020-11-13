// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_WGPU_OP_LRN_HPP
#define OPENCV_DNN_WGPU_OP_LRN_HPP

#include "webgpu_common.hpp"
#include "webgpu_op_base.hpp"
namespace cv { namespace dnn { namespace webgpu {

#ifdef HAVE_WEBGPU

enum LRNShaderType
{
    kLRNShaderTypeBasic = 0,
    kLRNShaderTypeNum
};

struct LRNShaderConfig
{
    int local_size_x;
    int local_size_y;
    int local_size_z;
    int block_height;
    int block_width;
    int block_depth;
    LRNShaderType shader_type;
};

class OpLRN : public OpBase
{
public:
    OpLRN(const int radius, const float bias,
          const float alpha, const float beta,
          const bool norm_by_size);
    void reshapeOutTensor(Tensor& in, Tensor& out);
    bool forward(Tensor& in, Tensor& out);
    virtual bool forward(std::vector<Tensor>& ins,
                         std::vector<Tensor>& blobs,
                         std::vector<Tensor>& outs) CV_OVERRIDE;

private:
    bool init(const int radius, const float bias,
              const float alpha, const float beta,
              const bool norm_by_size);
    bool computeGroupCount();
    int batch_;
    int height_;
    int width_;
    int channels_;
    int radius_;
    float bias_;
    float alpha_;
    float beta_;
    int filter_len_;
    int thread_num_;
    bool norm_by_size_;
    LRNShaderConfig config_;
};

#endif  // HAVE_WEBGPU

}}} // namespace cv::dnn::webgpu

#endif // OPENCV_DNN_WGPU_OP_LRN_HPP