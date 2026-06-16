// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "metal_private.hpp"

#ifdef HAVE_METAL

namespace cv {
namespace metal {
namespace {

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
#include "kernels/copy_to_mask.metal"
        ;

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

struct SubtractParams
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

struct MultiplyParams
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
    float scale;
};

struct BitwiseParams
{
    uint64_t src1Offset;
    uint64_t src2Offset;
    uint64_t dstOffset;
    uint64_t src1Step;
    uint64_t src2Step;
    uint64_t dstStep;
    int rows;
    int cols;
    int elemSize;
    int op;
};

struct CompareParams
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
    int op;
};

struct ConvertToParams
{
    uint64_t srcOffset;
    uint64_t dstOffset;
    uint64_t srcStep;
    uint64_t dstStep;
    int rows;
    int cols;
    int sdepth;
    int ddepth;
    int channels;
    float alpha;
    float beta;
};

struct ThresholdParams
{
    uint64_t srcOffset;
    uint64_t dstOffset;
    uint64_t srcStep;
    uint64_t dstStep;
    int rows;
    int cols;
    int depth;
    int channels;
    int thresholdType;
    float thresh;
    float maxval;
};

id<MTLComputePipelineState> getAddPipeline(const std::shared_ptr<MetalContext>& ctx)
{
    static id<MTLComputePipelineState> pipeline = nil;
    if (pipeline)
        return pipeline;

    static const char* source =
#include "kernels/add.metal"
        ;

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

id<MTLComputePipelineState> getSubtractPipeline(const std::shared_ptr<MetalContext>& ctx)
{
    static id<MTLComputePipelineState> pipeline = nil;
    if (pipeline)
        return pipeline;

    static const char* source =
#include "kernels/subtract.metal"
        ;

    NSError* error = nil;
    NSString* metalSource = [NSString stringWithUTF8String:source];
    id<MTLLibrary> library = [ctx->device() newLibraryWithSource:metalSource options:nil error:&error];
    if (!library)
        return nil;

    id<MTLFunction> function = [library newFunctionWithName:@"subtractKernel"];
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

id<MTLComputePipelineState> getMultiplyPipeline(const std::shared_ptr<MetalContext>& ctx)
{
    static id<MTLComputePipelineState> pipeline = nil;
    if (pipeline)
        return pipeline;

    static const char* source =
#include "kernels/multiply.metal"
        ;

    NSError* error = nil;
    NSString* metalSource = [NSString stringWithUTF8String:source];
    id<MTLLibrary> library = [ctx->device() newLibraryWithSource:metalSource options:nil error:&error];
    if (!library)
        return nil;

    id<MTLFunction> function = [library newFunctionWithName:@"multiplyKernel"];
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

id<MTLComputePipelineState> getBitwisePipeline(const std::shared_ptr<MetalContext>& ctx)
{
    static id<MTLComputePipelineState> pipeline = nil;
    if (pipeline)
        return pipeline;

    static const char* source =
#include "kernels/bitwise.metal"
        ;

    NSError* error = nil;
    NSString* metalSource = [NSString stringWithUTF8String:source];
    id<MTLLibrary> library = [ctx->device() newLibraryWithSource:metalSource options:nil error:&error];
    if (!library)
        return nil;

    id<MTLFunction> function = [library newFunctionWithName:@"bitwiseKernel"];
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

id<MTLComputePipelineState> getComparePipeline(const std::shared_ptr<MetalContext>& ctx)
{
    static id<MTLComputePipelineState> pipeline = nil;
    if (pipeline)
        return pipeline;

    static const char* source =
#include "kernels/compare.metal"
        ;

    NSError* error = nil;
    NSString* metalSource = [NSString stringWithUTF8String:source];
    id<MTLLibrary> library = [ctx->device() newLibraryWithSource:metalSource options:nil error:&error];
    if (!library)
        return nil;

    id<MTLFunction> function = [library newFunctionWithName:@"compareKernel"];
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

id<MTLComputePipelineState> getConvertToPipeline(const std::shared_ptr<MetalContext>& ctx)
{
    static id<MTLComputePipelineState> pipeline = nil;
    if (pipeline)
        return pipeline;

    static const char* source =
#include "kernels/convert_to.metal"
        ;

    NSError* error = nil;
    NSString* metalSource = [NSString stringWithUTF8String:source];
    id<MTLLibrary> library = [ctx->device() newLibraryWithSource:metalSource options:nil error:&error];
    if (!library)
        return nil;

    id<MTLFunction> function = [library newFunctionWithName:@"convertToKernel"];
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

id<MTLComputePipelineState> getThresholdPipeline(const std::shared_ptr<MetalContext>& ctx)
{
    static id<MTLComputePipelineState> pipeline = nil;
    if (pipeline)
        return pipeline;

    static const char* source =
#include "kernels/threshold.metal"
        ;

    NSError* error = nil;
    NSString* metalSource = [NSString stringWithUTF8String:source];
    id<MTLLibrary> library = [ctx->device() newLibraryWithSource:metalSource options:nil error:&error];
    if (!library)
        return nil;

    id<MTLFunction> function = [library newFunctionWithName:@"thresholdKernel"];
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

struct SetToParams
{
    uint64_t dstOffset;
    uint64_t maskOffset;
    uint64_t dstStep;
    uint64_t maskStep;
    int rows;
    int cols;
    int elemSize;
    int depthSize;
    int channels;
    int maskChannels;
    int haveMask;
    uint8_t scalar[16];
};

id<MTLComputePipelineState> getSetToPipeline(const std::shared_ptr<MetalContext>& ctx)
{
    static id<MTLComputePipelineState> pipeline = nil;
    if (pipeline)
        return pipeline;

    static const char* source =
#include "kernels/set_to.metal"
        ;

    NSError* error = nil;
    NSString* metalSource = [NSString stringWithUTF8String:source];
    id<MTLLibrary> library = [ctx->device() newLibraryWithSource:metalSource options:nil error:&error];
    if (!library)
        return nil;

    id<MTLFunction> function = [library newFunctionWithName:@"setToKernel"];
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

bool subtract(const UMat& src1, const UMat& src2, UMat& dst)
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

    id<MTLComputePipelineState> pipeline = getSubtractPipeline(ctx);
    if (!pipeline)
        return false;

    size_t src1Ofs[CV_MAX_DIM] = {0};
    size_t src2Ofs[CV_MAX_DIM] = {0};
    size_t dstOfs[CV_MAX_DIM] = {0};
    src1.ndoffset(src1Ofs);
    src2.ndoffset(src2Ofs);
    dst.ndoffset(dstOfs);

    SubtractParams params;
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

bool multiply(const UMat& src1, const UMat& src2, UMat& dst, double scale)
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

    id<MTLComputePipelineState> pipeline = getMultiplyPipeline(ctx);
    if (!pipeline)
        return false;

    size_t src1Ofs[CV_MAX_DIM] = {0};
    size_t src2Ofs[CV_MAX_DIM] = {0};
    size_t dstOfs[CV_MAX_DIM] = {0};
    src1.ndoffset(src1Ofs);
    src2.ndoffset(src2Ofs);
    dst.ndoffset(dstOfs);

    MultiplyParams params;
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
    params.scale = static_cast<float>(scale);

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

bool bitwise(const UMat& src1, const UMat& src2, UMat& dst, int op)
{
    if (!haveMetal() || src1.dims != 2 || src2.dims != 2 || dst.dims != 2)
        return false;
    if (src1.rows != src2.rows || src1.cols != src2.cols ||
        src1.rows != dst.rows || src1.cols != dst.cols ||
        src1.type() != src2.type() || src1.type() != dst.type())
        return false;
    if (op < 9 || op > 12)
        return false;

    MetalBuffer* src1Buffer = getBuffer(src1.u);
    MetalBuffer* src2Buffer = getBuffer(src2.u);
    MetalBuffer* dstBuffer = getBuffer(dst.u);
    if (!src1Buffer || !src2Buffer || !dstBuffer)
        return false;

    std::shared_ptr<MetalContext> ctx = std::static_pointer_cast<MetalContext>(dst.u->allocatorContext);
    if (!ctx || !ctx->valid())
        return false;

    id<MTLComputePipelineState> pipeline = getBitwisePipeline(ctx);
    if (!pipeline)
        return false;

    size_t src1Ofs[CV_MAX_DIM] = {0};
    size_t src2Ofs[CV_MAX_DIM] = {0};
    size_t dstOfs[CV_MAX_DIM] = {0};
    src1.ndoffset(src1Ofs);
    src2.ndoffset(src2Ofs);
    dst.ndoffset(dstOfs);

    BitwiseParams params;
    params.src1Offset = src1Ofs[0] * src1.step.p[0] + src1Ofs[1] * CV_ELEM_SIZE(src1.type());
    params.src2Offset = src2Ofs[0] * src2.step.p[0] + src2Ofs[1] * CV_ELEM_SIZE(src2.type());
    params.dstOffset = dstOfs[0] * dst.step.p[0] + dstOfs[1] * CV_ELEM_SIZE(dst.type());
    params.src1Step = src1.step.p[0];
    params.src2Step = src2.step.p[0];
    params.dstStep = dst.step.p[0];
    params.rows = src1.rows;
    params.cols = src1.cols;
    params.elemSize = static_cast<int>(CV_ELEM_SIZE(src1.type()));
    params.op = op;

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

bool compare(const UMat& src1, const UMat& src2, UMat& dst, int op)
{
    if (!haveMetal() || src1.dims != 2 || src2.dims != 2 || dst.dims != 2)
        return false;
    if (src1.rows != src2.rows || src1.cols != src2.cols ||
        src1.rows != dst.rows || src1.cols != dst.cols ||
        src1.type() != src2.type() || dst.type() != CV_8UC(src1.channels()))
        return false;

    int depth = CV_MAT_DEPTH(src1.type());
    int channels = src1.channels();
    if ((depth != CV_8U && depth != CV_32F) || (channels != 1 && channels != 3 && channels != 4))
        return false;
    if (op < 0 || op > 5)
        return false;

    MetalBuffer* src1Buffer = getBuffer(src1.u);
    MetalBuffer* src2Buffer = getBuffer(src2.u);
    MetalBuffer* dstBuffer = getBuffer(dst.u);
    if (!src1Buffer || !src2Buffer || !dstBuffer)
        return false;

    std::shared_ptr<MetalContext> ctx = std::static_pointer_cast<MetalContext>(dst.u->allocatorContext);
    if (!ctx || !ctx->valid())
        return false;

    id<MTLComputePipelineState> pipeline = getComparePipeline(ctx);
    if (!pipeline)
        return false;

    size_t src1Ofs[CV_MAX_DIM] = {0};
    size_t src2Ofs[CV_MAX_DIM] = {0};
    size_t dstOfs[CV_MAX_DIM] = {0};
    src1.ndoffset(src1Ofs);
    src2.ndoffset(src2Ofs);
    dst.ndoffset(dstOfs);

    CompareParams params;
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
    params.op = op;

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

bool convertTo(const UMat& src, UMat& dst, int ddepth, double alpha, double beta)
{
    if (!haveMetal() || src.dims != 2 || dst.dims != 2)
        return false;
    if (src.rows != dst.rows || src.cols != dst.cols || dst.depth() != ddepth ||
        src.channels() != dst.channels())
        return false;

    int sdepth = CV_MAT_DEPTH(src.type());
    int channels = src.channels();
    if ((sdepth != CV_8U && sdepth != CV_32F) || (ddepth != CV_8U && ddepth != CV_32F) ||
        (channels != 1 && channels != 3 && channels != 4))
        return false;

    MetalBuffer* srcBuffer = getBuffer(src.u);
    MetalBuffer* dstBuffer = getBuffer(dst.u);
    if (!srcBuffer || !dstBuffer)
        return false;

    std::shared_ptr<MetalContext> ctx = std::static_pointer_cast<MetalContext>(dst.u->allocatorContext);
    if (!ctx || !ctx->valid())
        return false;

    id<MTLComputePipelineState> pipeline = getConvertToPipeline(ctx);
    if (!pipeline)
        return false;

    size_t srcOfs[CV_MAX_DIM] = {0};
    size_t dstOfs[CV_MAX_DIM] = {0};
    src.ndoffset(srcOfs);
    dst.ndoffset(dstOfs);

    ConvertToParams params;
    params.srcOffset = srcOfs[0] * src.step.p[0] + srcOfs[1] * CV_ELEM_SIZE(src.type());
    params.dstOffset = dstOfs[0] * dst.step.p[0] + dstOfs[1] * CV_ELEM_SIZE(dst.type());
    params.srcStep = src.step.p[0];
    params.dstStep = dst.step.p[0];
    params.rows = src.rows;
    params.cols = src.cols;
    params.sdepth = sdepth;
    params.ddepth = ddepth;
    params.channels = channels;
    params.alpha = static_cast<float>(alpha);
    params.beta = static_cast<float>(beta);

    id<MTLCommandBuffer> commandBuffer = [ctx->queue() commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!commandBuffer || !encoder)
        return false;

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:srcBuffer->buffer offset:0 atIndex:0];
    [encoder setBuffer:dstBuffer->buffer offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];

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

bool threshold(const UMat& src, UMat& dst, double thresh, double maxval, int thresholdType)
{
    if (!haveMetal() || src.dims != 2 || dst.dims != 2)
        return false;
    if (src.rows != dst.rows || src.cols != dst.cols || src.type() != dst.type())
        return false;

    int depth = CV_MAT_DEPTH(src.type());
    int channels = src.channels();
    if ((depth != CV_8U && depth != CV_32F) || (channels != 1 && channels != 3 && channels != 4))
        return false;
    if (thresholdType < 0 || thresholdType > 4)
        return false;

    if (depth == CV_8U && (thresh < 0.0 || thresh >= 255.0))
        return false;

    MetalBuffer* srcBuffer = getBuffer(src.u);
    MetalBuffer* dstBuffer = getBuffer(dst.u);
    if (!srcBuffer || !dstBuffer)
        return false;

    std::shared_ptr<MetalContext> ctx = std::static_pointer_cast<MetalContext>(dst.u->allocatorContext);
    if (!ctx || !ctx->valid())
        return false;

    id<MTLComputePipelineState> pipeline = getThresholdPipeline(ctx);
    if (!pipeline)
        return false;

    size_t srcOfs[CV_MAX_DIM] = {0};
    size_t dstOfs[CV_MAX_DIM] = {0};
    src.ndoffset(srcOfs);
    dst.ndoffset(dstOfs);

    ThresholdParams params;
    params.srcOffset = srcOfs[0] * src.step.p[0] + srcOfs[1] * CV_ELEM_SIZE(src.type());
    params.dstOffset = dstOfs[0] * dst.step.p[0] + dstOfs[1] * CV_ELEM_SIZE(dst.type());
    params.srcStep = src.step.p[0];
    params.dstStep = dst.step.p[0];
    params.rows = src.rows;
    params.cols = src.cols;
    params.depth = depth;
    params.channels = channels;
    params.thresholdType = thresholdType;
    params.thresh = static_cast<float>(thresh);
    params.maxval = static_cast<float>(maxval);

    id<MTLCommandBuffer> commandBuffer = [ctx->queue() commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!commandBuffer || !encoder)
        return false;

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:srcBuffer->buffer offset:0 atIndex:0];
    [encoder setBuffer:dstBuffer->buffer offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];

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

bool setTo(UMat& dst, const Mat& value, const UMat* mask)
{
    if (!haveMetal() || dst.dims != 2)
        return false;
    if (mask && (mask->dims != 2 || mask->rows != dst.rows || mask->cols != dst.cols))
        return false;

    int depth = CV_MAT_DEPTH(dst.type());
    int channels = dst.channels();
    if ((depth != CV_8U && depth != CV_32F) || (channels != 1 && channels != 3 && channels != 4))
        return false;

    int maskChannels = 0;
    if (mask)
    {
        int maskDepth = CV_MAT_DEPTH(mask->type());
        maskChannels = CV_MAT_CN(mask->type());
        if ((maskDepth != CV_8U && maskDepth != CV_8S && maskDepth != CV_Bool) ||
            (maskChannels != 1 && maskChannels != channels))
            return false;
    }

    MetalBuffer* dstBuffer = getBuffer(dst.u);
    MetalBuffer* maskBuffer = mask ? getBuffer(mask->u) : NULL;
    if (!dstBuffer || (mask && !maskBuffer))
        return false;

    std::shared_ptr<MetalContext> ctx = std::static_pointer_cast<MetalContext>(dst.u->allocatorContext);
    if (!ctx || !ctx->valid())
        return false;

    id<MTLComputePipelineState> pipeline = getSetToPipeline(ctx);
    if (!pipeline)
        return false;

    size_t dstOfs[CV_MAX_DIM] = {0};
    size_t maskOfs[CV_MAX_DIM] = {0};
    dst.ndoffset(dstOfs);
    if (mask)
        mask->ndoffset(maskOfs);

    SetToParams params = {};
    params.dstOffset = dstOfs[0] * dst.step.p[0] + dstOfs[1] * CV_ELEM_SIZE(dst.type());
    params.maskOffset = mask ? maskOfs[0] * mask->step.p[0] + maskOfs[1] * CV_ELEM_SIZE(mask->type()) : 0;
    params.dstStep = dst.step.p[0];
    params.maskStep = mask ? mask->step.p[0] : 0;
    params.rows = dst.rows;
    params.cols = dst.cols;
    params.elemSize = static_cast<int>(CV_ELEM_SIZE(dst.type()));
    params.depthSize = static_cast<int>(CV_ELEM_SIZE1(dst.type()));
    params.channels = channels;
    params.maskChannels = maskChannels;
    params.haveMask = mask ? 1 : 0;
    convertAndUnrollScalar(value, dst.type(), params.scalar, 1);

    id<MTLCommandBuffer> commandBuffer = [ctx->queue() commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!commandBuffer || !encoder)
        return false;

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:dstBuffer->buffer offset:0 atIndex:0];
    [encoder setBuffer:maskBuffer ? maskBuffer->buffer : nil offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];

    MTLSize threadgroup = MTLSizeMake(16, 16, 1);
    MTLSize groups = MTLSizeMake((dst.cols + threadgroup.width - 1) / threadgroup.width,
                                 (dst.rows + threadgroup.height - 1) / threadgroup.height,
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


} // namespace metal
} // namespace cv

#endif // HAVE_METAL
