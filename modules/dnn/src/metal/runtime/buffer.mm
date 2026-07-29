#include "../../precomp.hpp"
#include "buffer.hpp"

#include "allocator.hpp"

#include <cstring>

namespace cv { namespace dnn { namespace metal {

static void checkRange(size_t capacity, size_t offset, size_t size)
{
    CV_CheckLE(offset, capacity, "Metal buffer offset exceeds capacity");
    CV_CheckLE(size, capacity - offset, "Metal buffer range exceeds capacity");
}

Buffer::Buffer(size_t size)
{
    buffer_ = Allocator::get().allocate(size);
    size_ = size;
}

Buffer::~Buffer()
{
    Allocator::get().free(buffer_);
}

std::shared_ptr<Buffer> Buffer::create(size_t size)
{
    return std::shared_ptr<Buffer>(new Buffer(size));
}

std::shared_ptr<Buffer> Buffer::createWithData(const void* data,
                                               size_t size)
{
    CV_CheckTrue(data != nullptr, "Metal buffer source data must not be null");
    std::shared_ptr<Buffer> buffer = create(size);
    buffer->upload(data, size);
    return buffer;
}

size_t Buffer::size() const
{
    return size_;
}

id<MTLBuffer> Buffer::native() const
{
    return buffer_;
}

void Buffer::upload(const void* data, size_t size, size_t offset)
{
    CV_CheckTrue(data != nullptr, "Metal buffer source data must not be null");
    checkRange(size_, offset, size);
    std::memcpy(static_cast<unsigned char*>([buffer_ contents]) + offset, data, size);
}

void Buffer::download(void* data, size_t size, size_t offset) const
{
    CV_CheckTrue(data != nullptr, "Metal buffer destination data must not be null");
    checkRange(size_, offset, size);
    std::memcpy(data, static_cast<const unsigned char*>([buffer_ contents]) + offset, size);
}

}}} // namespace cv::dnn::metal
