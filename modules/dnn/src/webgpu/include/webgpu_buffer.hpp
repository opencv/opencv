// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_WEBGPU_BUFFER_HPP
#define OPENCV_DNN_WEBGPU_BUFFER_HPP

#include <unistd.h>
#include <memory>
#if defined(__EMSCRIPTEN__) && defined(DAWN_EMSDK)
#include <webgpu/webgpu_cpp.h>
#include <emscripten.h>
#else
#ifdef HAVE_WEBGPU
#include <dawn/webgpu_cpp.h>
#endif  // HAVE_WEBGPU
#endif
namespace cv { namespace dnn { namespace webgpu {
#ifdef HAVE_WEBGPU
class Buffer
{
public:
    Buffer(const std::shared_ptr<wgpu::Device> device);
    Buffer(const std::shared_ptr<wgpu::Device> device,
           const void* data, size_t size,
           wgpu::BufferUsage usage = wgpu::BufferUsage::Storage |
           wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
    Buffer(const void* data, size_t size,
           wgpu::BufferUsage usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
    ~Buffer()
    {
        if(buffer_) buffer_.Release();
        if(gpuReadBuffer_) gpuReadBuffer_.Release();
    }
    wgpu::Buffer * getWebGPUBuffer() { return & buffer_; }
    wgpu::BufferUsage getBufferUsage() { return usage_;}
    void setBufferData(const void * data, size_t size);
    const void* MapReadAsyncAndWait();
    void unMap() { if(gpuReadBuffer_) gpuReadBuffer_.Unmap(); }
    size_t getSize() { return size_; }
private:
    Buffer();
    std::shared_ptr<wgpu::Device> device_;
    wgpu::Buffer buffer_;
    wgpu::Buffer gpuReadBuffer_;
    wgpu::BufferUsage usage_;
    size_t size_;
    const void* mappedData = nullptr;
};

#endif  // HAVE_WEBGPU

}}}  //namespace cv::dnn::webgpu

#endif  //OPENCV_DNN_WEBGPU_OP_BASE_HPP