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

struct CopyToMaskParams
{
    uint64_t srcOffset;
    uint64_t maskOffset;
    uint64_t dstOffset;
    uint64_t srcStep;
    uint64_t maskStep;
    uint64_t dstStep;
    int rows;
    int cols;
    int elemSize;
    int depthSize;
    int channels;
    int maskChannels;
    int haveDstUninit;
};

id<MTLComputePipelineState> getCopyToMaskPipeline(const std::shared_ptr<MetalContext>& ctx)
{
    static id<MTLComputePipelineState> pipeline = nil;
    if (pipeline)
        return pipeline;

    static const char* source =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct CopyToMaskParams\n"
        "{\n"
        "    ulong srcOffset;\n"
        "    ulong maskOffset;\n"
        "    ulong dstOffset;\n"
        "    ulong srcStep;\n"
        "    ulong maskStep;\n"
        "    ulong dstStep;\n"
        "    int rows;\n"
        "    int cols;\n"
        "    int elemSize;\n"
        "    int depthSize;\n"
        "    int channels;\n"
        "    int maskChannels;\n"
        "    int haveDstUninit;\n"
        "};\n"
        "kernel void copyToMaskKernel(device const uchar* src [[buffer(0)]],\n"
        "                             device const uchar* mask [[buffer(1)]],\n"
        "                             device uchar* dst [[buffer(2)]],\n"
        "                             constant CopyToMaskParams& p [[buffer(3)]],\n"
        "                             uint2 gid [[thread_position_in_grid]])\n"
        "{\n"
        "    if ((int)gid.x >= p.cols || (int)gid.y >= p.rows)\n"
        "        return;\n"
        "    ulong srcBase = p.srcOffset + (ulong)gid.y * p.srcStep + (ulong)gid.x * (ulong)p.elemSize;\n"
        "    ulong maskBase = p.maskOffset + (ulong)gid.y * p.maskStep + (ulong)gid.x * (ulong)p.maskChannels;\n"
        "    ulong dstBase = p.dstOffset + (ulong)gid.y * p.dstStep + (ulong)gid.x * (ulong)p.elemSize;\n"
        "    if (p.maskChannels == 1)\n"
        "    {\n"
        "        if (mask[maskBase])\n"
        "        {\n"
        "            for (int i = 0; i < p.elemSize; ++i)\n"
        "                dst[dstBase + i] = src[srcBase + i];\n"
        "        }\n"
        "        else if (p.haveDstUninit)\n"
        "        {\n"
        "            for (int i = 0; i < p.elemSize; ++i)\n"
        "                dst[dstBase + i] = 0;\n"
        "        }\n"
        "    }\n"
        "    else\n"
        "    {\n"
        "        for (int c = 0; c < p.channels; ++c)\n"
        "        {\n"
        "            ulong srcChannel = srcBase + (ulong)c * (ulong)p.depthSize;\n"
        "            ulong dstChannel = dstBase + (ulong)c * (ulong)p.depthSize;\n"
        "            if (mask[maskBase + c])\n"
        "            {\n"
        "                for (int i = 0; i < p.depthSize; ++i)\n"
        "                    dst[dstChannel + i] = src[srcChannel + i];\n"
        "            }\n"
        "            else if (p.haveDstUninit)\n"
        "            {\n"
        "                for (int i = 0; i < p.depthSize; ++i)\n"
        "                    dst[dstChannel + i] = 0;\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "}\n";

    NSError* error = nil;
    NSString* metalSource = [NSString stringWithUTF8String:source];
    id<MTLLibrary> library = [ctx->device() newLibraryWithSource:metalSource options:nil error:&error];
    if (!library)
        return nil;

    id<MTLFunction> function = [library newFunctionWithName:@"copyToMaskKernel"];
    if (!function)
    {
        [library release];
        return nil;
    }

    pipeline = [ctx->device() newComputePipelineStateWithFunction:function error:&error];
    [function release];
    [library release];
    return pipeline;
}

struct AddParams
{
    uint64_t src1Offset;
    uint64_t src2Offset;
    uint64_t dstOffset;
    uint64_t src1Step;
    uint64_t src2Step;
    uint64_t dstStep;
    int rows;
    int cols;
    int depth;
    int channels;
};

id<MTLComputePipelineState> getAddPipeline(const std::shared_ptr<MetalContext>& ctx)
{
    static id<MTLComputePipelineState> pipeline = nil;
    if (pipeline)
        return pipeline;

    static const char* source =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct AddParams\n"
        "{\n"
        "    ulong src1Offset;\n"
        "    ulong src2Offset;\n"
        "    ulong dstOffset;\n"
        "    ulong src1Step;\n"
        "    ulong src2Step;\n"
        "    ulong dstStep;\n"
        "    int rows;\n"
        "    int cols;\n"
        "    int depth;\n"
        "    int channels;\n"
        "};\n"
        "kernel void addKernel(device const uchar* src1 [[buffer(0)]],\n"
        "                      device const uchar* src2 [[buffer(1)]],\n"
        "                      device uchar* dst [[buffer(2)]],\n"
        "                      constant AddParams& p [[buffer(3)]],\n"
        "                      uint2 gid [[thread_position_in_grid]])\n"
        "{\n"
        "    if ((int)gid.x >= p.cols || (int)gid.y >= p.rows)\n"
        "        return;\n"
        "    if (p.depth == 0)\n"
        "    {\n"
        "        ulong src1Base = p.src1Offset + (ulong)gid.y * p.src1Step + (ulong)gid.x * (ulong)p.channels;\n"
        "        ulong src2Base = p.src2Offset + (ulong)gid.y * p.src2Step + (ulong)gid.x * (ulong)p.channels;\n"
        "        ulong dstBase = p.dstOffset + (ulong)gid.y * p.dstStep + (ulong)gid.x * (ulong)p.channels;\n"
        "        for (int c = 0; c < p.channels; ++c)\n"
        "        {\n"
        "            int sum = (int)src1[src1Base + c] + (int)src2[src2Base + c];\n"
        "            dst[dstBase + c] = (uchar)min(sum, 255);\n"
        "        }\n"
        "    }\n"
        "    else\n"
        "    {\n"
        "        ulong src1Base = p.src1Offset + (ulong)gid.y * p.src1Step + (ulong)gid.x * (ulong)p.channels * sizeof(float);\n"
        "        ulong src2Base = p.src2Offset + (ulong)gid.y * p.src2Step + (ulong)gid.x * (ulong)p.channels * sizeof(float);\n"
        "        ulong dstBase = p.dstOffset + (ulong)gid.y * p.dstStep + (ulong)gid.x * (ulong)p.channels * sizeof(float);\n"
        "        device const float* src1f = (device const float*)(src1 + src1Base);\n"
        "        device const float* src2f = (device const float*)(src2 + src2Base);\n"
        "        device float* dstf = (device float*)(dst + dstBase);\n"
        "        for (int c = 0; c < p.channels; ++c)\n"
        "            dstf[c] = src1f[c] + src2f[c];\n"
        "    }\n"
        "}\n";

    NSError* error = nil;
    NSString* metalSource = [NSString stringWithUTF8String:source];
    id<MTLLibrary> library = [ctx->device() newLibraryWithSource:metalSource options:nil error:&error];
    if (!library)
        return nil;

    id<MTLFunction> function = [library newFunctionWithName:@"addKernel"];
    if (!function)
    {
        [library release];
        return nil;
    }

    pipeline = [ctx->device() newComputePipelineStateWithFunction:function error:&error];
    [function release];
    [library release];
    return pipeline;
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

bool copyToMask(const UMat& src, const UMat& mask, UMat& dst, bool haveDstUninit)
{
    if (!haveMetal() || src.dims != 2 || mask.dims != 2 || dst.dims != 2)
        return false;
    if (src.rows != mask.rows || src.cols != mask.cols ||
        src.rows != dst.rows || src.cols != dst.cols || src.type() != dst.type())
        return false;

    int channels = src.channels();
    int maskChannels = CV_MAT_CN(mask.type());
    if (CV_MAT_DEPTH(mask.type()) != CV_8U || (maskChannels != 1 && maskChannels != channels))
        return false;

    MetalBuffer* srcBuffer = getBuffer(src.u);
    MetalBuffer* maskBuffer = getBuffer(mask.u);
    MetalBuffer* dstBuffer = getBuffer(dst.u);
    if (!srcBuffer || !maskBuffer || !dstBuffer)
        return false;

    std::shared_ptr<MetalContext> ctx = std::static_pointer_cast<MetalContext>(dst.u->allocatorContext);
    if (!ctx || !ctx->valid())
        return false;

    id<MTLComputePipelineState> pipeline = getCopyToMaskPipeline(ctx);
    if (!pipeline)
        return false;

    size_t srcOfs[CV_MAX_DIM] = {0};
    size_t maskOfs[CV_MAX_DIM] = {0};
    size_t dstOfs[CV_MAX_DIM] = {0};
    src.ndoffset(srcOfs);
    mask.ndoffset(maskOfs);
    dst.ndoffset(dstOfs);

    CopyToMaskParams params;
    params.srcOffset = srcOfs[0] * src.step.p[0] + srcOfs[1] * CV_ELEM_SIZE(src.type());
    params.maskOffset = maskOfs[0] * mask.step.p[0] + maskOfs[1] * CV_ELEM_SIZE(mask.type());
    params.dstOffset = dstOfs[0] * dst.step.p[0] + dstOfs[1] * CV_ELEM_SIZE(dst.type());
    params.srcStep = src.step.p[0];
    params.maskStep = mask.step.p[0];
    params.dstStep = dst.step.p[0];
    params.rows = src.rows;
    params.cols = src.cols;
    params.elemSize = static_cast<int>(CV_ELEM_SIZE(src.type()));
    params.depthSize = static_cast<int>(CV_ELEM_SIZE1(src.type()));
    params.channels = channels;
    params.maskChannels = maskChannels;
    params.haveDstUninit = haveDstUninit ? 1 : 0;

    id<MTLCommandBuffer> commandBuffer = [ctx->queue() commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!commandBuffer || !encoder)
        return false;

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:srcBuffer->buffer offset:0 atIndex:0];
    [encoder setBuffer:maskBuffer->buffer offset:0 atIndex:1];
    [encoder setBuffer:dstBuffer->buffer offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];

    MTLSize threadgroup = MTLSizeMake(16, 16, 1);
    MTLSize groups = MTLSizeMake((src.cols + threadgroup.width - 1) / threadgroup.width,
                                 (src.rows + threadgroup.height - 1) / threadgroup.height,
                                 1);
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threadgroup];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    if ([commandBuffer status] == MTLCommandBufferStatusError)
        return false;

    dst.u->markDeviceCopyObsolete(false);
    dst.u->markHostCopyObsolete(true);
    return true;
}

bool add(const UMat& src1, const UMat& src2, UMat& dst)
{
    if (!haveMetal() || src1.dims != 2 || src2.dims != 2 || dst.dims != 2)
        return false;
    if (src1.rows != src2.rows || src1.cols != src2.cols ||
        src1.rows != dst.rows || src1.cols != dst.cols ||
        src1.type() != src2.type() || src1.type() != dst.type())
        return false;

    int depth = CV_MAT_DEPTH(src1.type());
    int channels = src1.channels();
    if ((depth != CV_8U && depth != CV_32F) || (channels != 1 && channels != 3 && channels != 4))
        return false;

    MetalBuffer* src1Buffer = getBuffer(src1.u);
    MetalBuffer* src2Buffer = getBuffer(src2.u);
    MetalBuffer* dstBuffer = getBuffer(dst.u);
    if (!src1Buffer || !src2Buffer || !dstBuffer)
        return false;

    std::shared_ptr<MetalContext> ctx = std::static_pointer_cast<MetalContext>(dst.u->allocatorContext);
    if (!ctx || !ctx->valid())
        return false;

    id<MTLComputePipelineState> pipeline = getAddPipeline(ctx);
    if (!pipeline)
        return false;

    size_t src1Ofs[CV_MAX_DIM] = {0};
    size_t src2Ofs[CV_MAX_DIM] = {0};
    size_t dstOfs[CV_MAX_DIM] = {0};
    src1.ndoffset(src1Ofs);
    src2.ndoffset(src2Ofs);
    dst.ndoffset(dstOfs);

    AddParams params;
    params.src1Offset = src1Ofs[0] * src1.step.p[0] + src1Ofs[1] * CV_ELEM_SIZE(src1.type());
    params.src2Offset = src2Ofs[0] * src2.step.p[0] + src2Ofs[1] * CV_ELEM_SIZE(src2.type());
    params.dstOffset = dstOfs[0] * dst.step.p[0] + dstOfs[1] * CV_ELEM_SIZE(dst.type());
    params.src1Step = src1.step.p[0];
    params.src2Step = src2.step.p[0];
    params.dstStep = dst.step.p[0];
    params.rows = src1.rows;
    params.cols = src1.cols;
    params.depth = depth;
    params.channels = channels;

    id<MTLCommandBuffer> commandBuffer = [ctx->queue() commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!commandBuffer || !encoder)
        return false;

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:src1Buffer->buffer offset:0 atIndex:0];
    [encoder setBuffer:src2Buffer->buffer offset:0 atIndex:1];
    [encoder setBuffer:dstBuffer->buffer offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];

    MTLSize threadgroup = MTLSizeMake(16, 16, 1);
    MTLSize groups = MTLSizeMake((src1.cols + threadgroup.width - 1) / threadgroup.width,
                                 (src1.rows + threadgroup.height - 1) / threadgroup.height,
                                 1);
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threadgroup];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    if ([commandBuffer status] == MTLCommandBufferStatusError)
        return false;

    dst.u->markDeviceCopyObsolete(false);
    dst.u->markHostCopyObsolete(true);
    return true;
}

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
