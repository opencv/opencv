// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_VKCOM_TENSOR_HPP
#define OPENCV_DNN_VKCOM_TENSOR_HPP

#ifdef HAVE_VULKAN
#include <vulkan/vulkan.h>
#endif
#include <memory>
#include "vkcom.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

class Buffer;

class Tensor
{
public:
    Tensor(Format fmt = kFormatFp32, VkBufferUsageFlags usageFlag = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Tensor(const char* data, std::vector<int>& shape, Format fmt = kFormatFp32,
           VkBufferUsageFlags usageFlag = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    void* map();
    void unMap();
    Shape getShape() const;
    int dimSize(const int dim) const;
    int dimNum() const;
    int count(const int start_axis = 0, const int end_axis = -1) const;

    // Change shape and format to as passed in.
    // Copy data if data != NULL
    // Allocate new internal buffer if new size > old size or alloc flag is true
    Tensor reshape(const char* data, const std::vector<int>& shape, bool alloc = false, Format fmt = kFormatInvalid);

    void setTo(float val);
    int getFormat() const;
    size_t size() const { return size_in_byte_; }
    bool isEmpty() { return size_in_byte_ == 0 ? true : false; }
    void copyTo(Tensor& dst);
    Ptr<Buffer> getBuffer() { return buffer_; }

private:
    std::vector<int> shape_;
    size_t size_in_byte_;
    Ptr<Buffer> buffer_;
    Format format_;
    VkBufferUsageFlags usageFlag_;
};

#endif  // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom

#endif // OPENCV_DNN_VKCOM_TENSOR_HPP
