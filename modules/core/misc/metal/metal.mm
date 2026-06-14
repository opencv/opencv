// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../src/precomp.hpp"

#ifdef HAVE_METAL

#include "../../src/metal.hpp"
#include "../../src/umatrix.hpp"
#include "opencv2/core/metal.hpp"

#import <Metal/Metal.h>

namespace cv {
namespace metal {
namespace {

struct MetalBuffer
{
    MetalBuffer(id<MTLBuffer> buffer_, size_t size_, bool hostVisible_)
        : buffer(buffer_), size(size_), hostVisible(hostVisible_)
    {
    }

    ~MetalBuffer()
    {
        [buffer release];
    }

    id<MTLBuffer> buffer;
    size_t size;
    bool hostVisible;
};

class MetalContext
{
public:
    MetalContext()
    {
        device_ = MTLCreateSystemDefaultDevice();
        if (device_)
            queue_ = [device_ newCommandQueue];
    }

    ~MetalContext()
    {
        [queue_ release];
        [device_ release];
    }

    bool valid() const
    {
        return device_ != nil && queue_ != nil;
    }

    id<MTLDevice> device() const
    {
        return device_;
    }

    id<MTLCommandQueue> queue() const
    {
        return queue_;
    }

private:
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
};

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

class MetalAllocator CV_FINAL : public MatAllocator
{
public:
    UMatData* allocate(int dims, const int* sizes, int type,
                       void* data, size_t* step, AccessFlag, UMatUsageFlags usageFlags) const CV_OVERRIDE
    {
        if (!haveMetal())
            return Mat::getDefaultAllocator()->allocate(dims, sizes, type, data, step, ACCESS_RW, USAGE_DEFAULT);

        CV_Assert(data == NULL);

        size_t total = CV_ELEM_SIZE(type);
        for (int i = dims - 1; i >= 0; i--)
        {
            if (step)
                step[i] = total;
            total *= sizes[i];
        }

        std::shared_ptr<MetalContext> ctx = getMetalContext();
        bool hostVisible = true;
        MTLResourceOptions storageOptions = getStorageOptions(usageFlags, hostVisible);
        id<MTLBuffer> buffer = [ctx->device() newBufferWithLength:total options:storageOptions];
        if (!buffer)
            return Mat::getDefaultAllocator()->allocate(dims, sizes, type, data, step, ACCESS_RW, USAGE_DEFAULT);

        UMatData* u = new UMatData(this);
        u->size = total;
        u->handle = new MetalBuffer(buffer, total, hostVisible);
        u->allocatorContext = ctx;
        u->markHostCopyObsolete(true);
        return u;
    }

    bool allocate(UMatData* u, AccessFlag accessFlags, UMatUsageFlags usageFlags) const CV_OVERRIDE
    {
        if (!u || !haveMetal())
            return false;

        UMatDataAutoLock lock(u);
        if (u->handle)
            return true;

        CV_Assert(u->origdata != NULL);

        std::shared_ptr<MetalContext> ctx = getMetalContext();
        bool hostVisible = true;
        MTLResourceOptions storageOptions = getStorageOptions(usageFlags, hostVisible);
        id<MTLBuffer> buffer = [ctx->device() newBufferWithLength:u->size options:storageOptions];
        if (!buffer)
            return false;

        u->handle = new MetalBuffer(buffer, u->size, hostVisible);
        u->prevAllocator = u->currAllocator;
        u->currAllocator = this;
        u->allocatorContext = ctx;
        u->flags |= UMatData::TEMP_UMAT | UMatData::TEMP_COPIED_UMAT;

        uploadFromHost(u, u->origdata);
        u->markDeviceCopyObsolete(false);
        u->markHostCopyObsolete(!!(accessFlags & ACCESS_WRITE));
        return true;
    }

    void deallocate(UMatData* u) const CV_OVERRIDE
    {
        if (!u)
            return;

        CV_Assert(u->urefcount == 0);
        CV_Assert(u->refcount == 0);
        CV_Assert(u->mapcount == 0);

        if (u->tempUMat())
        {
            if (u->hostCopyObsolete() && u->origdata)
            {
                downloadToHost(u, u->origdata);
                u->markHostCopyObsolete(false);
            }

            delete getBuffer(u);
            u->handle = NULL;
            u->markDeviceCopyObsolete(true);
            u->currAllocator = u->prevAllocator;
            u->prevAllocator = NULL;
            u->data = u->origdata;
            u->currAllocator->deallocate(u);
            return;
        }

        delete getBuffer(u);
        u->handle = NULL;
        u->markDeviceCopyObsolete(true);
        delete u;
    }

    void map(UMatData* u, AccessFlag accessFlags) const CV_OVERRIDE
    {
        CV_Assert(u && u->handle);

        if (!!(accessFlags & ACCESS_WRITE))
            u->markDeviceCopyObsolete(true);

        if (!u->deviceMemMapped())
        {
            CV_Assert(u->mapcount++ == 0);
            if (isHostVisible(u))
            {
                u->data = getContents(u);
            }
            else
            {
                if (!u->tempUMat())
                    u->data = static_cast<uchar*>(fastMalloc(u->size));
                else
                    u->data = u->origdata;
                CV_Assert(u->data);
                if (!!(accessFlags & ACCESS_READ) && u->hostCopyObsolete())
                    downloadToHost(u, u->data);
            }
            u->markDeviceMemMapped(true);
        }

        u->markHostCopyObsolete(false);
    }

    void unmap(UMatData* u) const CV_OVERRIDE
    {
        if (!u)
            return;

        if (u->deviceMemMapped())
        {
            CV_Assert(u->mapcount > 0);
            u->mapcount--;
            if (!isHostVisible(u) && u->deviceCopyObsolete())
            {
                uploadFromHost(u, u->data);
            }
            u->markDeviceMemMapped(false);
            u->markDeviceCopyObsolete(false);
            u->markHostCopyObsolete(false);
            if (!isHostVisible(u) && !u->tempUMat())
            {
                fastFree(u->data);
                u->data = NULL;
            }
            else if (!u->tempUMat())
            {
                u->data = NULL;
            }
            else if (!isHostVisible(u))
            {
                u->data = u->origdata;
            }
        }

        if (u->urefcount == 0 && u->refcount == 0)
            deallocate(u);
    }

    void download(UMatData* u, void* dst, int dims, const size_t sz[],
                  const size_t srcofs[], const size_t srcstep[],
                  const size_t dststep[]) const CV_OVERRIDE
    {
        if (!u)
            return;
        uchar* oldData = u->data;
        id<MTLBuffer> staging = nil;
        if (isHostVisible(u))
        {
            u->data = getContents(u);
        }
        else
        {
            std::shared_ptr<MetalContext> ctx = std::static_pointer_cast<MetalContext>(u->allocatorContext);
            CV_Assert(ctx && ctx->valid());
            staging = newSharedBuffer(ctx, u->size);
            CV_Assert(staging);
            blitCopy(ctx, getBuffer(u)->buffer, 0, staging, 0, u->size);
            u->data = static_cast<uchar*>([staging contents]);
        }
        MatAllocator::download(u, dst, dims, sz, srcofs, srcstep, dststep);
        u->data = oldData;
        if (staging)
            [staging release];
        u->markHostCopyObsolete(false);
    }

    void upload(UMatData* u, const void* src, int dims, const size_t sz[],
                const size_t dstofs[], const size_t dststep[],
                const size_t srcstep[]) const CV_OVERRIDE
    {
        if (!u)
            return;
        uchar* oldData = u->data;
        id<MTLBuffer> staging = nil;
        std::shared_ptr<MetalContext> ctx;
        if (isHostVisible(u))
        {
            u->data = getContents(u);
        }
        else
        {
            ctx = std::static_pointer_cast<MetalContext>(u->allocatorContext);
            CV_Assert(ctx && ctx->valid());
            staging = newSharedBuffer(ctx, u->size);
            CV_Assert(staging);
            blitCopy(ctx, getBuffer(u)->buffer, 0, staging, 0, u->size);
            u->data = static_cast<uchar*>([staging contents]);
        }
        MatAllocator::upload(u, src, dims, sz, dstofs, dststep, srcstep);
        if (staging)
        {
            blitCopy(ctx, staging, 0, getBuffer(u)->buffer, 0, u->size);
            [staging release];
        }
        u->data = oldData;
        u->markDeviceCopyObsolete(false);
        u->markHostCopyObsolete(true);
    }

    void copy(UMatData* src, UMatData* dst, int dims, const size_t sz[],
              const size_t srcofs[], const size_t srcstep[],
              const size_t dstofs[], const size_t dststep[], bool) const CV_OVERRIDE
    {
        if (!src || !dst)
            return;

        size_t srcrawofs = 0, dstrawofs = 0, total = 0, dstTotal = 0;
        bool srcContinuous = isContinuousRegion(dims, sz, srcofs, srcstep, srcrawofs, total);
        bool dstContinuous = isContinuousRegion(dims, sz, dstofs, dststep, dstrawofs, dstTotal);

        MetalBuffer* srcBuffer = getBuffer(src);
        MetalBuffer* dstBuffer = getBuffer(dst);
        std::shared_ptr<MetalContext> ctx = std::static_pointer_cast<MetalContext>(dst->allocatorContext);
        if (srcContinuous && dstContinuous && total == dstTotal && srcBuffer && dstBuffer && ctx && ctx->valid())
        {
            blitCopy(ctx, srcBuffer->buffer, srcrawofs, dstBuffer->buffer, dstrawofs, total);
        }
        else
        {
            uchar* oldSrcData = src->data;
            uchar* oldDstData = dst->data;
            id<MTLBuffer> srcStaging = nil;
            id<MTLBuffer> dstStaging = nil;

            if (isHostVisible(src))
            {
                src->data = getContents(src);
            }
            else
            {
                std::shared_ptr<MetalContext> srcCtx = std::static_pointer_cast<MetalContext>(src->allocatorContext);
                CV_Assert(srcCtx && srcCtx->valid());
                srcStaging = newSharedBuffer(srcCtx, src->size);
                CV_Assert(srcStaging);
                blitCopy(srcCtx, srcBuffer->buffer, 0, srcStaging, 0, src->size);
                src->data = static_cast<uchar*>([srcStaging contents]);
            }

            if (isHostVisible(dst))
            {
                dst->data = getContents(dst);
            }
            else
            {
                CV_Assert(ctx && ctx->valid());
                dstStaging = newSharedBuffer(ctx, dst->size);
                CV_Assert(dstStaging);
                blitCopy(ctx, dstBuffer->buffer, 0, dstStaging, 0, dst->size);
                dst->data = static_cast<uchar*>([dstStaging contents]);
            }

            MatAllocator::copy(src, dst, dims, sz, srcofs, srcstep, dstofs, dststep, true);
            if (dstStaging)
            {
                blitCopy(ctx, dstStaging, 0, dstBuffer->buffer, 0, dst->size);
                [dstStaging release];
            }
            if (srcStaging)
                [srcStaging release];
            src->data = oldSrcData;
            dst->data = oldDstData;
        }

        dst->markDeviceCopyObsolete(false);
        dst->markHostCopyObsolete(true);
    }
};

MetalAllocator* getMetalAllocator_()
{
    return new MetalAllocator();
}

} // namespace

bool haveMetal()
{
    return getMetalContext()->valid();
}

MatAllocator* getMetalAllocator()
{
    CV_SINGLETON_LAZY_INIT(MatAllocator, getMetalAllocator_())
}

} // namespace metal
} // namespace cv

#endif // HAVE_METAL
