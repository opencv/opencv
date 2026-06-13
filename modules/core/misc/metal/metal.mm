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
    MetalBuffer(id<MTLBuffer> buffer_, size_t size_)
        : buffer(buffer_), size(size_)
    {
    }

    ~MetalBuffer()
    {
        [buffer release];
    }

    id<MTLBuffer> buffer;
    size_t size;
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
    return b ? static_cast<uchar*>([b->buffer contents]) : NULL;
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
                       void* data, size_t* step, AccessFlag, UMatUsageFlags) const CV_OVERRIDE
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
        id<MTLBuffer> buffer = [ctx->device() newBufferWithLength:total options:MTLResourceStorageModeShared];
        if (!buffer)
            return Mat::getDefaultAllocator()->allocate(dims, sizes, type, data, step, ACCESS_RW, USAGE_DEFAULT);

        UMatData* u = new UMatData(this);
        u->size = total;
        u->handle = new MetalBuffer(buffer, total);
        u->allocatorContext = ctx;
        u->markHostCopyObsolete(true);
        return u;
    }

    bool allocate(UMatData* u, AccessFlag accessFlags, UMatUsageFlags) const CV_OVERRIDE
    {
        if (!u || !haveMetal())
            return false;

        UMatDataAutoLock lock(u);
        if (u->handle)
            return true;

        CV_Assert(u->origdata != NULL);

        std::shared_ptr<MetalContext> ctx = getMetalContext();
        id<MTLBuffer> buffer = [ctx->device() newBufferWithLength:u->size options:MTLResourceStorageModeShared];
        if (!buffer)
            return false;

        u->handle = new MetalBuffer(buffer, u->size);
        u->prevAllocator = u->currAllocator;
        u->currAllocator = this;
        u->allocatorContext = ctx;
        u->flags |= UMatData::TEMP_UMAT | UMatData::TEMP_COPIED_UMAT;

        memcpy(getContents(u), u->origdata, u->size);
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
                memcpy(u->origdata, getContents(u), u->size);
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
            u->data = getContents(u);
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
            u->markDeviceMemMapped(false);
            u->markDeviceCopyObsolete(false);
            u->markHostCopyObsolete(false);
            if (!u->tempUMat())
                u->data = NULL;
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
        u->data = getContents(u);
        MatAllocator::download(u, dst, dims, sz, srcofs, srcstep, dststep);
        u->data = oldData;
        u->markHostCopyObsolete(false);
    }

    void upload(UMatData* u, const void* src, int dims, const size_t sz[],
                const size_t dstofs[], const size_t dststep[],
                const size_t srcstep[]) const CV_OVERRIDE
    {
        if (!u)
            return;
        uchar* oldData = u->data;
        u->data = getContents(u);
        MatAllocator::upload(u, src, dims, sz, dstofs, dststep, srcstep);
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
            id<MTLCommandBuffer> commandBuffer = [ctx->queue() commandBuffer];
            id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
            [blit copyFromBuffer:srcBuffer->buffer sourceOffset:srcrawofs
                        toBuffer:dstBuffer->buffer destinationOffset:dstrawofs
                            size:total];
            [blit endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
        }
        else
        {
            uchar* oldSrcData = src->data;
            uchar* oldDstData = dst->data;
            src->data = getContents(src);
            dst->data = getContents(dst);
            MatAllocator::copy(src, dst, dims, sz, srcofs, srcstep, dstofs, dststep, true);
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
