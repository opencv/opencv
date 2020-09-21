// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_WEBGPU_TENSOR_HPP
#define OPENCV_DNN_WEBGPU_TENSOR_HPP
#if defined(__EMSCRIPTEN__) && defined(DAWN_EMSDK)
#include <webgpu/webgpu_cpp.h>
#else
#ifdef HAVE_WEBGPU
#include <dawn/webgpu_cpp.h>
#endif  // HAVE_WEBGPU
#endif
#include "webgpu_common.hpp"
#include <memory>
namespace cv { namespace dnn { namespace webgpu {
#ifdef HAVE_WEBGPU
class Buffer;
class Tensor{
public:
    Tensor(Format fmt = wFormatFp32);
    Tensor(const void* data, std::vector<int>& shape,
           Format fmt = wFormatFp32);
    const void* mapRead();
    void unMap();
    Shape getShape() const;
    int dimSize(const int dim) const;
    int dimNum() const;
    int count(const int start_axis = 0, const int end_axis = -1) const;
    // Change shape and format to as passed in.
    // Copy data if data != NULL
    // Allocate new internal buffer if new size > old size or alloc flag is true
    Tensor reshape(const void* data, const std::vector<int>& shape,
                   bool alloc = false,
                   Format fmt = wFormatFp32);
    int getFormat() const;
    size_t size() const { return size_in_byte_; }
    bool isEmpty() { return size_in_byte_ == 0 ? true : false; }
    void copyTo(Tensor& dst);
    std::shared_ptr<Buffer> getBuffer() { return buffer_; }
private:
    std::shared_ptr<wgpu::Device> device_;
    std::vector<int> shape_;
    size_t size_in_byte_;
    std::shared_ptr<Buffer> buffer_;
    Format format_ = wFormatFp32;
    wgpu::BufferUsage usage_ = wgpu::BufferUsage::Storage |
    wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
};

#endif  // HAVE_WEBGPU

}}}  //namespace cv::dnn:webgpu

#endif  //  OPENCV_DNN_WEBGPU_TENSOR_HPP