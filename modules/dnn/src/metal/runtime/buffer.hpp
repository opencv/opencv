#ifndef OPENCV_DNN_METAL_RUNTIME_BUFFER_HPP
#define OPENCV_DNN_METAL_RUNTIME_BUFFER_HPP

#import <Metal/Metal.h>

#include <cstddef>
#include <memory>

namespace cv { namespace dnn { namespace metal {

class Buffer
{
public:
    static std::shared_ptr<Buffer> create(size_t size);
    static std::shared_ptr<Buffer> createWithData(const void* data, size_t size);

    ~Buffer();

    size_t size() const;
    id<MTLBuffer> native() const;

    void upload(const void* data, size_t size, size_t offset = 0);
    void download(void* data, size_t size, size_t offset = 0) const;

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&&) = delete;
    Buffer& operator=(Buffer&&) = delete;

private:
    explicit Buffer(size_t size);

    id<MTLBuffer> buffer_ = nil;
    size_t size_ = 0;
};

}}}  // namespace cv::dnn::metal

#endif  // OPENCV_DNN_METAL_RUNTIME_BUFFER_HPP
