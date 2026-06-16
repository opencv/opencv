// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "metal_private.hpp"
#include "opencv2/core/utils/configuration.private.hpp"

#include <list>

#ifdef HAVE_METAL

namespace cv {
namespace metal {
namespace {

size_t allocationGranularity(size_t size)
{
    if (size < 1024 * 1024)
        return 4096;
    if (size < 16 * 1024 * 1024)
        return 64 * 1024;
    return 1024 * 1024;
}

class MetalBufferPool CV_FINAL : public BufferPoolController
{
public:
    MetalBufferPool()
        : maxReservedSize_(utils::getConfigurationParameterSizeT("OPENCV_METAL_BUFFERPOOL_LIMIT", 64 * 1024 * 1024))
    {
    }

    ~MetalBufferPool()
    {
        freeAllReservedBuffers();
    }

    MetalBuffer* allocate(const std::shared_ptr<MetalContext>& ctx, size_t size,
                          MTLResourceOptions storageOptions, bool hostVisible)
    {
        AutoLock lock(mutex_);
        size_t capacity = alignSize(size, (int)allocationGranularity(size));
        for (std::list<MetalBuffer*>::iterator it = reserved_.begin(); it != reserved_.end(); ++it)
        {
            MetalBuffer* candidate = *it;
            if (candidate->storageOptions != storageOptions)
                continue;
            if (candidate->size < size)
                continue;

            size_t diff = candidate->size - size;
            if (diff >= std::max((size_t)4096, size / 8))
                continue;

            reservedSize_ -= candidate->size;
            reserved_.erase(it);
            return candidate;
        }

        id<MTLBuffer> buffer = [ctx->device() newBufferWithLength:capacity options:storageOptions];
        if (!buffer)
            return NULL;
        return new MetalBuffer(buffer, capacity, hostVisible, storageOptions);
    }

    void release(MetalBuffer* buffer)
    {
        if (!buffer)
            return;

        AutoLock lock(mutex_);
        if (maxReservedSize_ == 0 || buffer->size > maxReservedSize_ / 8)
        {
            delete buffer;
            return;
        }

        reserved_.push_front(buffer);
        reservedSize_ += buffer->size;
        trim();
    }

    size_t getReservedSize() const CV_OVERRIDE
    {
        AutoLock lock(mutex_);
        return reservedSize_;
    }

    size_t getMaxReservedSize() const CV_OVERRIDE
    {
        AutoLock lock(mutex_);
        return maxReservedSize_;
    }

    void setMaxReservedSize(size_t size) CV_OVERRIDE
    {
        AutoLock lock(mutex_);
        maxReservedSize_ = size;
        trim();
    }

    void freeAllReservedBuffers() CV_OVERRIDE
    {
        AutoLock lock(mutex_);
        for (MetalBuffer* buffer : reserved_)
            delete buffer;
        reserved_.clear();
        reservedSize_ = 0;
    }

private:
    void trim()
    {
        for (std::list<MetalBuffer*>::iterator it = reserved_.begin(); it != reserved_.end();)
        {
            MetalBuffer* buffer = *it;
            if (buffer->size > maxReservedSize_ / 8)
            {
                reservedSize_ -= buffer->size;
                delete buffer;
                it = reserved_.erase(it);
            }
            else
            {
                ++it;
            }
        }

        while (reservedSize_ > maxReservedSize_ && !reserved_.empty())
        {
            MetalBuffer* buffer = reserved_.back();
            reservedSize_ -= buffer->size;
            delete buffer;
            reserved_.pop_back();
        }
    }

    mutable Mutex mutex_;
    size_t reservedSize_ = 0;
    size_t maxReservedSize_;
    std::list<MetalBuffer*> reserved_;
};

MetalBufferPool& getMetalBufferPool()
{
    static MetalBufferPool pool;
    return pool;
}

} // namespace

MetalBuffer::MetalBuffer(id<MTLBuffer> buffer_, size_t size_, bool hostVisible_, MTLResourceOptions storageOptions_)
    : buffer(buffer_), size(size_), hostVisible(hostVisible_), storageOptions(storageOptions_)
{
}

MetalBuffer::~MetalBuffer()
{
    [buffer release];
}

MetalContext::MetalContext()
{
    device_ = MTLCreateSystemDefaultDevice();
    if (device_)
        queue_ = [device_ newCommandQueue];
}

MetalContext::~MetalContext()
{
    [queue_ release];
    [device_ release];
}

bool MetalContext::valid() const
{
    return device_ != nil && queue_ != nil;
}

id<MTLDevice> MetalContext::device() const
{
    return device_;
}

id<MTLCommandQueue> MetalContext::queue() const
{
    return queue_;
}

std::shared_ptr<MetalContext> getMetalContext()
{
    static std::shared_ptr<MetalContext> ctx = std::make_shared<MetalContext>();
    return ctx;
}

MetalBuffer* getBuffer(UMatData* u)
{
    return u ? static_cast<MetalBuffer*>(u->handle) : NULL;
}

uchar* getContents(UMatData* u)
{
    MetalBuffer* b = getBuffer(u);
    return b && b->hostVisible ? static_cast<uchar*>([b->buffer contents]) : NULL;
}

MTLResourceOptions getStorageOptions(UMatUsageFlags usageFlags, bool& hostVisible)
{
    if ((usageFlags & USAGE_ALLOCATE_DEVICE_MEMORY) != 0 &&
        (usageFlags & (USAGE_ALLOCATE_HOST_MEMORY | USAGE_ALLOCATE_SHARED_MEMORY)) == 0)
    {
        hostVisible = false;
        return MTLResourceStorageModePrivate;
    }

    hostVisible = true;
    return MTLResourceStorageModeShared;
}

MetalBuffer* allocateBuffer(const std::shared_ptr<MetalContext>& ctx, size_t size,
                            MTLResourceOptions storageOptions, bool hostVisible)
{
    return getMetalBufferPool().allocate(ctx, size, storageOptions, hostVisible);
}

void releaseBuffer(MetalBuffer* buffer)
{
    getMetalBufferPool().release(buffer);
}

BufferPoolController* getMetalBufferPoolController()
{
    return &getMetalBufferPool();
}

bool isHostVisible(UMatData* u)
{
    MetalBuffer* b = getBuffer(u);
    return b && b->hostVisible;
}

id<MTLBuffer> newSharedBuffer(const std::shared_ptr<MetalContext>& ctx, size_t size)
{
    return [ctx->device() newBufferWithLength:size options:MTLResourceStorageModeShared];
}

void blitCopy(const std::shared_ptr<MetalContext>& ctx,
              id<MTLBuffer> src, size_t srcofs,
              id<MTLBuffer> dst, size_t dstofs,
              size_t size)
{
    if (size == 0)
        return;

    id<MTLCommandBuffer> commandBuffer = [ctx->queue() commandBuffer];
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    [blit copyFromBuffer:src sourceOffset:srcofs toBuffer:dst destinationOffset:dstofs size:size];
    [blit endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

void downloadToHost(UMatData* u, void* dst)
{
    MetalBuffer* b = getBuffer(u);
    CV_Assert(b);

    if (b->hostVisible)
    {
        memcpy(dst, [b->buffer contents], u->size);
        return;
    }

    std::shared_ptr<MetalContext> ctx = std::static_pointer_cast<MetalContext>(u->allocatorContext);
    CV_Assert(ctx && ctx->valid());

    id<MTLBuffer> staging = newSharedBuffer(ctx, u->size);
    CV_Assert(staging);
    blitCopy(ctx, b->buffer, 0, staging, 0, u->size);
    memcpy(dst, [staging contents], u->size);
    [staging release];
}

void uploadFromHost(UMatData* u, const void* src)
{
    MetalBuffer* b = getBuffer(u);
    CV_Assert(b);

    if (b->hostVisible)
    {
        memcpy([b->buffer contents], src, u->size);
        return;
    }

    std::shared_ptr<MetalContext> ctx = std::static_pointer_cast<MetalContext>(u->allocatorContext);
    CV_Assert(ctx && ctx->valid());

    id<MTLBuffer> staging = newSharedBuffer(ctx, u->size);
    CV_Assert(staging);
    memcpy([staging contents], src, u->size);
    blitCopy(ctx, staging, 0, b->buffer, 0, u->size);
    [staging release];
}

bool isContinuousRegion(int dims, const size_t sz[], const size_t ofs[],
                        const size_t step[], size_t& rawofs, size_t& total)
{
    if (dims <= 0)
        return false;

    rawofs = 0;
    total = sz[dims - 1];
    for (int i = 0; i < dims; i++)
    {
        if (sz[i] == 0)
            return false;
        if (ofs)
            rawofs += ofs[i] * (i <= dims - 2 ? step[i] : 1);
    }

    for (int i = dims - 2; i >= 0; i--)
    {
        if (step[i] != total)
            return false;
        total *= sz[i];
    }

    return true;
}


} // namespace metal
} // namespace cv

#endif // HAVE_METAL
