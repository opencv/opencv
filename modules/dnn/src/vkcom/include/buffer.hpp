// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_VKCOM_BUFFER_HPP
#define OPENCV_DNN_VKCOM_BUFFER_HPP

#ifdef HAVE_VULKAN
#include <vulkan/vulkan.h>
#endif // HAVE_VULKAN

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

class Buffer
{
public:
    Buffer(VkDevice& device)
        : device_(device), buffer_(VK_NULL_HANDLE), memory_(VK_NULL_HANDLE){};
    Buffer(VkDevice& device, size_t size_in_bytes, const char* data);
    ~Buffer();
    VkDeviceMemory getVkMemory() { return memory_; }
    VkBuffer getVkBuffer() { return buffer_; }

private:
    Buffer();
    bool init(size_t size_in_bytes, const char* data);
    VkDevice device_;
    VkBuffer buffer_;
    VkDeviceMemory memory_;
};

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom

#endif // OPENCV_DNN_VKCOM_BUFFER_HPP
