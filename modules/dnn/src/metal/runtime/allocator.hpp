#ifndef OPENCV_DNN_METAL_RUNTIME_ALLOCATOR_HPP
#define OPENCV_DNN_METAL_RUNTIME_ALLOCATOR_HPP

#import <Metal/Metal.h>

#include <cstddef>
#include <memory>
#include <mutex>

namespace cv { namespace dnn { namespace metal {

class Device;

class Allocator
{
public:
    static Allocator& get();

    id<MTLBuffer> allocate(size_t size);
    void free(id<MTLBuffer>& buffer);

    Allocator(const Allocator&) = delete;
    Allocator& operator=(const Allocator&) = delete;
    Allocator(Allocator&&) = delete;
    Allocator& operator=(Allocator&&) = delete;

private:
    Allocator();
    ~Allocator();

    static const size_t SMALL_BUFFER_SIZE = 256;
    static const size_t HEAP_SIZE = 1 << 20;

    std::shared_ptr<Device> device_;
    id<MTLHeap> heap_ = nil;
    std::mutex mutex_;
};

}}}  // namespace cv::dnn::metal

#endif  // OPENCV_DNN_METAL_RUNTIME_ALLOCATOR_HPP
