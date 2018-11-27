// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_VKCOM_OP_BASE_HPP
#define OPENCV_DNN_VKCOM_OP_BASE_HPP

#include "../../precomp.hpp"
#include "vkcom.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

// Forward declare
class Context;

class OpBase
{
public:
    OpBase();
    virtual ~OpBase();
    virtual bool forward(std::vector<Tensor>& ins,
                         std::vector<Tensor>& blobs,
                         std::vector<Tensor>& outs) = 0;
protected:
    void initVulkanThing(int buffer_num);
    void createDescriptorSetLayout(int buffer_num);
    void createDescriptorSet(int buffer_num);
    void createShaderModule(const uint32_t* spv, size_t sz, const std::string& source = std::string());
    void createPipeline(size_t push_constants_size = 0);
    void createCommandBuffer();
    void recordCommandBuffer(void* push_constants = NULL, size_t push_constants_size = 0);
    void runCommandBuffer();

    VkPipeline pipeline_;
    VkCommandBuffer cmd_buffer_;
    VkDescriptorPool descriptor_pool_;
    VkDescriptorSet descriptor_set_;
    VkDevice device_;
    VkDescriptorSetLayout descriptor_set_layout_;
    VkPipelineLayout pipeline_layout_;
    VkShaderModule module_;
    int group_x_;
    int group_y_;
    int group_z_;
    std::string type_;
};

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom

#endif // OPENCV_DNN_VKCOM_OP_BASE_HPP
