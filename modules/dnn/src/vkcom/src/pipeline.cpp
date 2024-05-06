// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
The code has referenced MNN (https://github.com/alibaba/MNN/blob/2.4.0/source/backend/vulkan/component/VulkanPipeline.cpp)
and adapted for OpenCV by Zihao Mu.
Below is the original copyright:
*/

//
//  VulkanPipeline.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../../precomp.hpp"
#include "internal.hpp"
#include "../include/pipeline.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

// *********************** Descriptor ********************
Ptr<Descriptor> Descriptor::create(const VkDescriptorPool& pool, const VkDescriptorSet& set, Pipeline* pipeline)
{
    return Ptr<Descriptor>(new Descriptor(pool, set, pipeline));
}

Descriptor::Descriptor(const VkDescriptorPool& pool, const VkDescriptorSet& set, Pipeline* _pipeline)
: desPool(pool), desSet(set), pipeline(_pipeline)
{
}

void Descriptor::writeTensor(Tensor tensor, int bindIndex)
{
    writeBuffer(tensor.getBuffer()->getVkBuffer(), bindIndex, tensor.size()); // TODO, check if need the size in bit.
}

void Descriptor::writeBuffer(VkBuffer buffer, int bindIndex, size_t size, VkDeviceSize offset)
{
    CV_Assert(pipeline);
    VkWriteDescriptorSet writeSet = {};
    VkDescriptorBufferInfo sourceInfo;
    sourceInfo.buffer        = buffer;
    sourceInfo.offset        = offset;
    sourceInfo.range         = size;

    writeSet.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeSet.descriptorCount = 1;

    writeSet.descriptorType  = pipeline->argType(bindIndex);
    writeSet.dstBinding      = bindIndex;
    writeSet.pBufferInfo     = &sourceInfo;
    writeSet.dstSet          = desSet;

    vkUpdateDescriptorSets(kDevice, 1, &writeSet, 0, nullptr);
}

Descriptor::~Descriptor()
{
    if (needRelease)
    {
        // destroy resource
        vkFreeDescriptorSets(kDevice, desPool, 1, &desSet);
        vkDestroyDescriptorPool(kDevice, desPool, nullptr);
    }
    else
    {
        CV_Assert(pipeline);
        pipeline->descriptorPairQueue.push(std::make_pair(desPool, desSet));
    }
}

// *********************** Pipeline ********************

Pipeline::Pipeline(const uint32_t* spv, size_t length,
                   const std::vector<VkDescriptorType>& _bufferTypes, VkPipelineCache& cache,
                   const std::vector<uint32_t>& localSize) : bufferTypes(_bufferTypes)
{
    // Step1: create Module from spv file.
    // TODO, add the local_size_x, local_size_y, and z at here.
    VkShaderModule shaderModule;
    VkShaderModuleCreateInfo shaderModuleCreateInfo
    {
            /* .sType    = */ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            /* .pNext    = */ nullptr,
            /* .flags    = */ 0,
            /* .codeSize = */ length * sizeof(uint32_t),
            /* .pCode    = */ spv,
    };
    VK_CHECK_RESULT(vkCreateShaderModule(kDevice, &shaderModuleCreateInfo, nullptr, &shaderModule));

    // Step2: according the bufferType info set the binding.
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    std::map<VkDescriptorType, int> typeCount;

    for (int i = 0; i < bufferTypes.size(); i++)
    {
        auto type = bufferTypes[i];
        if (typeCount.find(type) == typeCount.end())
            typeCount[type] = 1;
        else
            typeCount[type] += 1;

        VkDescriptorSetLayoutBinding binding{(uint32_t)i, type, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings.emplace_back(binding);
    }

    // Step3 : Create DescriptorSetLayout and PipelineLayout
    {
        // Create DescriptorSetLayout
        VkDescriptorSetLayoutCreateInfo setLayoutCreateInfo = {};
        setLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        setLayoutCreateInfo.bindingCount = bindings.size();
        setLayoutCreateInfo.pBindings = &bindings[0];
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(kDevice, &setLayoutCreateInfo, NULL, &setLayout));

        // Create PipelineLayout
        VkPipelineLayoutCreateInfo pipeline_layout_create_info = {};
        pipeline_layout_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_create_info.setLayoutCount = 1;
        pipeline_layout_create_info.pSetLayouts = &setLayout;
        VK_CHECK_RESULT(vkCreatePipelineLayout(kDevice, &pipeline_layout_create_info, NULL, &pipelineLayout));
    }

    //Step: 4 create pipelineVk instance.
    VkPipelineShaderStageCreateInfo stageCreateInfo = {};
    stageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageCreateInfo.module = shaderModule;
    stageCreateInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stage = stageCreateInfo;
    pipelineCreateInfo.layout = pipelineLayout;

    cv::AutoLock lock(kContextMtx);
    VK_CHECK_RESULT(vkCreateComputePipelines(kDevice, cache, 1, &pipelineCreateInfo, 0, &pipelineVK));

    // Step5: destroy shaderModule
    vkDestroyShaderModule(kDevice, shaderModule, nullptr);

    // Step6: add typeCount to desPoolSize
    for (auto& iter : typeCount)
    {
        VkDescriptorPoolSize s;
        s.descriptorCount = iter.second;
        s.type            = iter.first;
        desPoolSize.emplace_back(s);
    }
}

Ptr<Pipeline> Pipeline::create(const uint32_t* spv, size_t length, const std::vector<VkDescriptorType>& bufferTypes,
                               VkPipelineCache& cache, const std::vector<uint32_t>& localSize)
{
    return Ptr<Pipeline>(new Pipeline(spv, length, bufferTypes, cache, localSize));
}

void Pipeline::bind(VkCommandBuffer cmdBuffer, VkDescriptorSet descriptorSet) const
{
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineVK);
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
}

Ptr<Descriptor> Pipeline::createSet()
{
    // find unused DescriptorSet at descriptorSetList, if not, create new one and save it at that list.
    if (!descriptorPairQueue.empty())
    {
        auto iter = descriptorPairQueue.front();
        descriptorPairQueue.pop();

        Ptr<Descriptor> des = Descriptor::create(iter.first, iter.second, this);
        des->needRelease = false; // Don't release and try to reuse it.
        return des;
    }

    // create DescriptorPool
    VkDescriptorPool descriptorPool;
    VkDescriptorPoolCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info.maxSets = 1;
    info.poolSizeCount = desPoolSize.size();
    info.pPoolSizes = desPoolSize.data();
    info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    VK_CHECK_RESULT(vkCreateDescriptorPool(kDevice, &info, NULL, &descriptorPool));

    // Create DescriptorSet
    VkDescriptorSet descriptorSet;
    VkDescriptorSetAllocateInfo allocate_info = {};
    allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocate_info.descriptorPool = descriptorPool;
    allocate_info.descriptorSetCount = 1;
    allocate_info.pSetLayouts = &setLayout;
    VK_CHECK_RESULT(vkAllocateDescriptorSets(kDevice, &allocate_info, &descriptorSet));

    Ptr<Descriptor> descriptor = Descriptor::create(descriptorPool, descriptorSet, this);
    descriptor->needRelease = false; // Don't release and try to reuse it.
    return descriptor;
}

Pipeline::~Pipeline()
{
    // Step1: destroy all descriptors in descriptorSetList.
    while (!descriptorPairQueue.empty())
    {
        auto iter = descriptorPairQueue.front();
        descriptorPairQueue.pop();

        CV_Assert(iter.first && iter.second);
        vkFreeDescriptorSets(kDevice, iter.first, 1, &iter.second);
        vkDestroyDescriptorPool(kDevice, iter.first, nullptr);
    }

    // Step2: destroy other resources.
    vkDestroyPipelineLayout(kDevice, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(kDevice, setLayout, nullptr);
    vkDestroyPipeline(kDevice, pipelineVK, nullptr);
}

// *********************** Pipeline Factory ********************

static VkResult createPipelineCache(VkPipelineCache& pipelineCache)
{
    VkPipelineCacheCreateInfo pipelineCacheInfo {
            /* .sType           = */ VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
            /* .pNext           = */ nullptr,
            /* .flags           = */ 0, // reserved, must be 0
            /* .initialDataSize = */ 0,
            /* .pInitialData    = */ nullptr,
    };
    return vkCreatePipelineCache(kDevice, &pipelineCacheInfo, nullptr, &pipelineCache);
}

PipelineFactory::PipelineFactory()
{
    initSPVMaps(); // create maps from spv name to SPV file.
    // create PipelineCache
    VK_CHECK_RESULT(createPipelineCache(pipelineCache));
}

PipelineFactory::~PipelineFactory()
{
    pipelineCreated.clear();
    vkDestroyPipelineCache(kDevice, pipelineCache, nullptr);
}

void PipelineFactory::reset()
{
    // Step1: destroy old pipelineCache.
    vkDestroyPipelineCache(kDevice, pipelineCache, nullptr);

    // Step2: create new PipelineCache
    VK_CHECK_RESULT(createPipelineCache(pipelineCache));

    auto iter = pipelineCreated.begin();
    for (int i = 0; i < pipelineCreated.size(); i++, iter++)
    {
        iter->second.release();
    }

    pipelineCreated.clear();
}

Ptr<Pipeline> PipelineFactory::getPipeline(const std::string& key, const std::vector<VkDescriptorType>& types,
                            const std::vector<uint32_t>& localSize)
{
    auto iter = pipelineCreated.find(key);
    if (iter != pipelineCreated.end())
    {
        return iter->second;
    }

    // retrieve spv from SPVMaps with given key
    auto iterSPV  = SPVMaps.find(key);
    if (iterSPV == SPVMaps.end())
        CV_Error(cv::Error::StsError, "Can not create SPV with the given name:"+key+"!");

    const uint32_t* spv = iterSPV->second.first;
    size_t length = iterSPV->second.second;

    Ptr<Pipeline> pipeline = Pipeline::create(spv, length, types, pipelineCache, localSize);

    if (pipeline)
    {
        pipelineCreated.insert(std::make_pair(key, pipeline));
    }
    else
    {
        CV_Error(cv::Error::StsError, "Can not Created the VkPipeline "+key);
    }

    return pipeline;
}

Ptr<PipelineFactory> PipelineFactory::create()
{
    Ptr<PipelineFactory> pipelineFactory = Ptr<PipelineFactory>(new PipelineFactory());
    return pipelineFactory;
}

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom
