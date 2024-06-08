// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_FENCE_HPP
#define OPENCV_FENCE_HPP

#include "../../precomp.hpp"

#ifdef HAVE_VULKAN
#include <vulkan/vulkan.h>
#endif // HAVE_VULKAN

namespace cv { namespace dnn { namespace vkcom {
#ifdef HAVE_VULKAN
// Used for synchronize and wait
class Fence
{
public:
    Fence();
    ~Fence();

    VkFence get() const;
    VkResult reset() const;
    VkResult wait() const;

private:
    VkFence fence;
};
#endif // HAVE_VULKAN
}}} // namespace cv::dnn::vkcom

#endif //OPENCV_FENCE_HPP
