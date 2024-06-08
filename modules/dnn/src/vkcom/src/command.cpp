// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
The code has referenced MNN (https://github.com/alibaba/MNN/blob/2.4.0/source/backend/vulkan/component/VulkanCommandPool.cpp)
and adapted for OpenCV by Zihao Mu.
Below is the original copyright:
*/

//
//  VulkanCommandPool.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../../precomp.hpp"
#include "internal.hpp"
#include "../include/command.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

// *********************** CommandBuffer ********************
CommandBuffer::CommandBuffer(CommandPool* pool) : cmdPool(pool)
{
    CV_Assert(cmdPool);
    if (pool->bufferQueue.empty())
    {
        VkCommandBufferAllocateInfo cmdBufferCreateInfo {
                /* .sType              = */ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                /* .pNext              = */ nullptr,
                /* .commandPool        = */ cmdPool->get(),
                /* .level              = */ VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                /* .commandBufferCount = */ 1,
        };
        vkAllocateCommandBuffers(kDevice, &cmdBufferCreateInfo, &cmdBuffer);
    }
    else
    {
        cmdBuffer = pool->bufferQueue.front();
        pool-> bufferQueue.pop();
    }
}

void CommandBuffer::barrierSource(VkBuffer source, size_t start, size_t size, BarrierType type) const
{
    VkBufferMemoryBarrier barrier;
    barrier.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.buffer              = source;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.offset              = start;
    barrier.pNext               = nullptr;
    barrier.size                = size;
    switch (type) {
        case READ_WRITE:
            barrier.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
            break;
        case WRITE_WRITE:
            barrier.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            break;
        default:
            break;
    }
    vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1,
                         &barrier, 0, nullptr);
}

void CommandBuffer::beginRecord(VkCommandBufferUsageFlags flag)
{
    cv::AutoLock lock(kContextMtx);
    VkCommandBufferBeginInfo cmdBufferBeginInfo{
            /* .sType            = */ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            /* .pNext            = */ nullptr,
            /* .flags            = */ flag,
            /* .pInheritanceInfo = */ nullptr,
    };
    vkResetCommandBuffer(cmdBuffer, 0);

    VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufferBeginInfo));
}

void CommandBuffer::endRecord()
{
    VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
}

CommandBuffer::~CommandBuffer()
{
    CV_Assert(cmdPool);
    if (needRelease)
    {
        vkFreeCommandBuffers(kDevice, cmdPool->get(), 1, &cmdBuffer);
    }
    else
    {
        cmdPool->bufferQueue.push(cmdBuffer);
    }
}

// *********************** CommandPool ********************
Ptr<CommandPool> CommandPool::create(const VkQueue &q, uint32_t _queueFamilyIndex)
{
    cv::AutoLock lock(kContextMtx);
    Ptr<CommandPool> cmdPoolInstance = Ptr<CommandPool>(new CommandPool(q, _queueFamilyIndex));

    return cmdPoolInstance;
}

CommandPool::CommandPool(const VkQueue& q, uint32_t _queueFamilyIndex) : queue(q), cmdPool(VK_NULL_HANDLE), queueFamilyIndex(_queueFamilyIndex)
{
    cv::AutoLock lock(kContextMtx);
    VkCommandPoolCreateInfo cmdPoolCreateInfo{
        /* .sType            = */ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        /* .pNext            = */ nullptr,
        /* .flags            = */ VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        /* .queueFamilyIndex = */ queueFamilyIndex,
    };
    vkCreateCommandPool(kDevice, &cmdPoolCreateInfo, nullptr, &cmdPool);
}

void CommandPool::reset()
{
    // reset all bufferQueue.
    while (!bufferQueue.empty())
    {
        auto cmdBuffer = bufferQueue.front();
        bufferQueue.pop();

        vkFreeCommandBuffers(kDevice, cmdPool, 1, &cmdBuffer);
    }
}

CommandPool::~CommandPool()
{
    while (!bufferQueue.empty())
    {
        auto cmdBuffer = bufferQueue.front();
        bufferQueue.pop();

        vkFreeCommandBuffers(kDevice, cmdPool, 1, &cmdBuffer);
    }
    vkDestroyCommandPool(kDevice, cmdPool, nullptr);
}

Ptr<CommandBuffer> CommandPool::allocBuffer()
{
    auto cmdBuffer = Ptr<CommandBuffer>(new CommandBuffer(this));
    cmdBuffer->needRelease = false;
    return cmdBuffer;
}

void CommandPool::submitAndWait(VkCommandBuffer& _buffer) const
{
    auto buffer = _buffer;
    Fence fence = Fence();
    VkFence fenceVk = fence.get();
    VkSubmitInfo submit_info = {
            /* .sType                = */ VK_STRUCTURE_TYPE_SUBMIT_INFO,
            /* .pNext                = */ nullptr,
            /* .waitSemaphoreCount   = */ 0,
            /* .pWaitSemaphores      = */ nullptr,
            /* .pWaitDstStageMask    = */ nullptr,
            /* .commandBufferCount   = */ 1,
            /* .pCommandBuffers      = */ &buffer,
            /* .signalSemaphoreCount = */ 0,
            /* .pSignalSemaphores    = */ nullptr};
    // need the queue class.
    VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submit_info, fenceVk));
    VK_CHECK_RESULT(fence.wait());
}

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom