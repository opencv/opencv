// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../../precomp.hpp"
#include "common.hpp"
#include "internal.hpp"
#include "../include/op_base.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

OpBase::OpBase()
{
    createContext();
    device_ = kDevice;
    pipeline_ = VK_NULL_HANDLE;
    cmd_buffer_ = VK_NULL_HANDLE;
    descriptor_pool_ = VK_NULL_HANDLE;
    descriptor_set_ = VK_NULL_HANDLE;
    descriptor_set_layout_ = VK_NULL_HANDLE;
    pipeline_layout_ = VK_NULL_HANDLE;
    module_ = VK_NULL_HANDLE;
}

OpBase::~OpBase()
{
    vkDestroyShaderModule(device_, module_, NULL);
    vkDestroyDescriptorPool(device_, descriptor_pool_, NULL);
    vkDestroyDescriptorSetLayout(device_, descriptor_set_layout_, NULL);
    vkDestroyPipeline(device_, pipeline_, NULL);
    vkDestroyPipelineLayout(device_, pipeline_layout_, NULL);
}

void OpBase::initVulkanThing(int buffer_num)
{
    createDescriptorSetLayout(buffer_num);
    createDescriptorSet(buffer_num);
    createCommandBuffer();
}

void OpBase::createDescriptorSetLayout(int buffer_num)
{
    if (buffer_num <= 0)
        return;
    std::vector<VkDescriptorSetLayoutBinding> bindings(buffer_num);
    for (int i = 0; i < buffer_num; i++)
    {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.bindingCount = buffer_num;
    info.pBindings = &bindings[0];
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device_, &info, NULL, &descriptor_set_layout_));
}

void OpBase::createDescriptorSet(int buffer_num)
{
    VkDescriptorPoolSize pool_size = {};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = buffer_num;

    VkDescriptorPoolCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info.maxSets = 1;
    info.poolSizeCount = 1;
    info.pPoolSizes = &pool_size;
    VK_CHECK_RESULT(vkCreateDescriptorPool(device_, &info, NULL, &descriptor_pool_));

    VkDescriptorSetAllocateInfo allocate_info = {};
    allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocate_info.descriptorPool = descriptor_pool_;
    allocate_info.descriptorSetCount = 1;
    allocate_info.pSetLayouts = &descriptor_set_layout_;
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device_, &allocate_info, &descriptor_set_));
}

void OpBase::createShaderModule(const uint32_t* spv, size_t sz, const std::string& source)
{
    VkShaderModuleCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    if (spv)
    {
        create_info.pCode = spv;
        create_info.codeSize = sz;
    }
    else
    {
        // online compilation
        std::vector<uint32_t> code;
        code = compile("shader", shaderc_compute_shader, source);
        create_info.pCode = code.data();
        create_info.codeSize = sizeof(uint32_t) * code.size();
    }
    VK_CHECK_RESULT(vkCreateShaderModule(device_, &create_info, NULL, &module_));
}

void OpBase::createPipeline(size_t push_constants_size)
{
    // create pipeline
    VkPipelineShaderStageCreateInfo stage_create_info = {};
    stage_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_create_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_create_info.module = module_;
    stage_create_info.pName = "main";
    VkPushConstantRange push_constant_ranges[1] = {};
    push_constant_ranges[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant_ranges[0].offset = 0;
    push_constant_ranges[0].size = push_constants_size;

    VkPipelineLayoutCreateInfo pipeline_layout_create_info = {};
    pipeline_layout_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    if (push_constants_size != 0)
    {
        pipeline_layout_create_info.pushConstantRangeCount = 1;
        pipeline_layout_create_info.pPushConstantRanges = push_constant_ranges;
    }
    pipeline_layout_create_info.setLayoutCount = 1;
    pipeline_layout_create_info.pSetLayouts = &descriptor_set_layout_;
    VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipeline_layout_create_info,
                                           NULL, &pipeline_layout_));

    VkComputePipelineCreateInfo pipeline_create_info = {};
    pipeline_create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_create_info.stage = stage_create_info;
    pipeline_create_info.layout = pipeline_layout_;
    VK_CHECK_RESULT(vkCreateComputePipelines(device_, VK_NULL_HANDLE,
                                             1, &pipeline_create_info,
                                             NULL, &pipeline_));
}

void OpBase::createCommandBuffer()
{
    VkCommandBufferAllocateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    info.commandPool = kCmdPool;
    info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    info.commandBufferCount = 1;
    VK_CHECK_RESULT(vkAllocateCommandBuffers(device_, &info, &cmd_buffer_));
}

void OpBase::recordCommandBuffer(void* push_constants, size_t push_constants_size)
{
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    cv::AutoLock lock(kContextMtx);
    VK_CHECK_RESULT(vkBeginCommandBuffer(cmd_buffer_, &beginInfo));
    if (push_constants)
        vkCmdPushConstants(cmd_buffer_, pipeline_layout_,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           push_constants_size, push_constants);
    vkCmdBindPipeline(cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout_, 0, 1, &descriptor_set_, 0, NULL);
    vkCmdDispatch(cmd_buffer_, group_x_, group_y_, group_z_);

    VK_CHECK_RESULT(vkEndCommandBuffer(cmd_buffer_));
}

void OpBase::runCommandBuffer()
{
    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd_buffer_;

    VkFence fence;
    VkFenceCreateInfo fence_create_info_ = {};
    fence_create_info_.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_create_info_.flags = 0;

    VK_CHECK_RESULT(vkCreateFence(device_, &fence_create_info_, NULL, &fence));
    {
        cv::AutoLock lock(kContextMtx);
        VK_CHECK_RESULT(vkQueueSubmit(kQueue, 1, &submit_info, fence));
    }
    VK_CHECK_RESULT(vkWaitForFences(device_, 1, &fence, VK_TRUE, 100000000000));
    vkDestroyFence(device_, fence, NULL);
}

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom
