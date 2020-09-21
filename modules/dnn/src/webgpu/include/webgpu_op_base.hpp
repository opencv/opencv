// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_WEBGPU_OP_BASE_HPP
#define OPENCV_DNN_WEBGPU_OP_BASE_HPP

#include "../../precomp.hpp"
#include "webgpu_common.hpp"
#include "../dawn/dawn_utils.hpp"

namespace cv { namespace dnn { namespace webgpu {
#ifdef HAVE_WEBGPU
class Context;
class Tensor;
class OpBase
{
public:
    OpBase();
    virtual ~OpBase();
    virtual bool forward(std::vector<Tensor>& ins,
                         std::vector<Tensor>& blobs,
                         std::vector<Tensor>& outs) = 0;

protected:
    void createBindGroupLayout(int buffer_num);
    void createBindGroup();
    void createShaderModule(const uint32_t* spv,
                            uint32_t size,
                            const std::string& source = std::string());
    void createComputePipeline();
    void createCommandBuffer();
    void runCommandBuffer();

    std::shared_ptr<wgpu::Device> device_;
    wgpu::ComputePipeline pipeline_ = nullptr;
    wgpu::CommandBuffer cmd_buffer_= nullptr;
    wgpu::BindGroupLayout bindgrouplayout_= nullptr;
    wgpu::BindGroup bindgroup_= nullptr;
    wgpu::ShaderModule module_= nullptr;
    wgpu::PipelineLayout pipeline_layout_= nullptr;
    std::vector<wgpu::BindGroupEntry> bgEntries = {};
    Buffer* uniformBuffer_ = nullptr;

    uint32_t group_x_;
    uint32_t group_y_;
    uint32_t group_z_;
    std::string type_;
};

#endif  // HAVE_WEBGPU

}}}  //namespace cv::dnn::webgpu

#endif  //OPENCV_DNN_WEBGPU_OP_BASE_HPP