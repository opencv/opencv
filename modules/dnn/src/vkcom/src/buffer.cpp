// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../../precomp.hpp"
#include "internal.hpp"
#include "../include/buffer.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

static uint32_t findMemoryType(uint32_t memoryTypeBits, VkMemoryPropertyFlags properties)
{
    for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; ++i)
    {
        if ((memoryTypeBits & (1 << i)) &&
                ((physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & properties) == properties))
            return i;
    }
    return uint32_t(-1);
}

Buffer::Buffer(VkBufferUsageFlags usageFlag) : usageFlag_(usageFlag), buffer_(VK_NULL_HANDLE), memory_(VK_NULL_HANDLE)
{
}

bool Buffer::init(size_t size_in_bytes, const char* data)
{
    if (buffer_ != VK_NULL_HANDLE)
    {
        CV_LOG_WARNING(NULL, "Warn: Buffer object already inited!");
        return false;
    }

    VkBufferCreateInfo bufferCreateInfo = {};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = (VkDeviceSize)size_in_bytes;
    bufferCreateInfo.usage = usageFlag_;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK_RESULT(vkCreateBuffer(kDevice, &bufferCreateInfo, NULL, &buffer_));

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(kDevice, buffer_, &memoryRequirements);

    VkMemoryAllocateInfo allocateInfo = {};
    allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize = memoryRequirements.size;

    // TODO: Try to optimize the memory at discrete graphics card. For AMD and GPU discrete graphics card,
    //  we should use VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT.

    allocateInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits,
                                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                                  );
    VK_CHECK_RESULT(vkAllocateMemory(kDevice, &allocateInfo, NULL, &memory_));

    if (data)
    {
        char* dst;
        VK_CHECK_RESULT(vkMapMemory(kDevice, memory_, 0, size_in_bytes, 0, (void **)&dst));
        memcpy(dst, data, size_in_bytes);
        vkUnmapMemory(kDevice, memory_);
    }

    VK_CHECK_RESULT(vkBindBufferMemory(kDevice, buffer_, memory_, 0));
    return true;
}

Buffer::Buffer(size_t size_in_bytes, const char* data, VkBufferUsageFlags usageFlag) : usageFlag_(usageFlag)
{
    buffer_ = VK_NULL_HANDLE;
    memory_ = VK_NULL_HANDLE;
    init(size_in_bytes, data);
}

Buffer::~Buffer()
{
    vkFreeMemory(kDevice, memory_, NULL);
    vkDestroyBuffer(kDevice, buffer_, NULL);
}

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom
