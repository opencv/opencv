// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_COMMAND_VULKAN_HPP
#define OPENCV_COMMAND_VULKAN_HPP

#include <queue>
#ifdef HAVE_VULKAN
#include <vulkan/vulkan.h>
#endif // HAVE_VULKAN

#include "fence.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

class CommandPool;
// CommandBuffer will record and dispatch the VkCommand, it was allocated from CommandPool.
class CommandBuffer
{
public:
    ~CommandBuffer();

    void beginRecord(VkCommandBufferUsageFlags flag = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    void endRecord();

    enum BarrierType {
        READ_WRITE = 0,
        WRITE_WRITE = 1,
    };
    void barrierSource(VkBuffer source, size_t start, size_t size, BarrierType type = READ_WRITE) const;

    VkCommandBuffer get()
    {
        return cmdBuffer;
    }

private:
    friend class CommandPool;
    CommandBuffer(CommandPool* pool);

    CommandPool* cmdPool;
    VkCommandBuffer cmdBuffer;
    // If is true, the deconstructor will release the instance, otherwise, re-use it.
    bool needRelease = true;
};

class CommandPool
{
public:
    static Ptr<CommandPool> create(const VkQueue& q, uint32_t _queueFamilyIndex);

    void operator=(const CommandPool &) = delete;
    CommandPool(CommandPool &other) = delete;

    void reset();
    ~CommandPool();
    VkCommandPool get() const
    {
        return cmdPool;
    }

    Ptr<CommandBuffer> allocBuffer();
    void submitAndWait(VkCommandBuffer& buffer) const;

    std::queue<VkCommandBuffer > bufferQueue; // For re-use the CommandBuffer.

private:
    CommandPool(const VkQueue& q, uint32_t _queueFamilyIndex);
    const VkQueue& queue;
    VkCommandPool cmdPool;
    uint32_t queueFamilyIndex;
};

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom


#endif //OPENCV_COMMAND_VULKAN_HPP
