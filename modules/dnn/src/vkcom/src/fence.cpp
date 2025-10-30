// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "internal.hpp"
#include "../include/fence.hpp"

namespace cv { namespace dnn { namespace vkcom {
#ifdef HAVE_VULKAN

Fence::Fence()
{
    VkFenceCreateInfo fci{
            /* .sType = */ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            /* .pNext = */ nullptr,
            /* .flags = */ 0,
    };
    vkCreateFence(kDevice, &fci, nullptr, &fence);
}

VkFence Fence::get() const
{
    return fence;
}

VkResult Fence::reset() const
{
    return vkResetFences(kDevice, 1, &fence);
}

VkResult Fence::wait() const
{
    auto status = VK_TIMEOUT;

    do {
        status = vkWaitForFences(kDevice, 1, &fence, VK_TRUE, 5000000000);
    } while (status == VK_TIMEOUT);

    return status;
}

Fence::~Fence()
{
    vkDestroyFence(kDevice, fence, nullptr);
}

#endif // HAVE_VULKAN
}}} // namespace cv::dnn::vkcom