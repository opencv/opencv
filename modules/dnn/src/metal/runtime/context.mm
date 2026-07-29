#include "../../precomp.hpp"
#include "context.hpp"

#include "buffer.hpp"
#include "device.hpp"

#include <opencv2/core/utils/logger.hpp>

#include <limits>

#import <Foundation/Foundation.h>

namespace cv { namespace dnn { namespace metal {

static const size_t MAX_OPERATIONS_PER_COMMAND_BUFFER = 40;
static const size_t MAX_RESOURCE_BYTES_PER_COMMAND_BUFFER = 40 * 1024 * 1024;

Context::Context(const std::shared_ptr<Device>& device)
    : device_(device)
{
    @autoreleasepool
    {
        CV_CheckTrue(device_ != nullptr, "Metal device is unavailable");
        commandQueue_ = [device_->native() newCommandQueue];
        CV_CheckTrue(commandQueue_ != nil, "Failed to create Metal command queue");
    }
}

Context::~Context()
{
    try
    {
        synchronize();
    }
    catch (const cv::Exception& e)
    {
        CV_LOG_ERROR(NULL, cv::format("Metal synchronization failed during context teardown. %s",
                                      e.what()));
    }

#if !__has_feature(objc_arc)
    [commandQueue_ release];
#endif
}

id<MTLComputeCommandEncoder> Context::computeEncoder()
{
    ensureCommandBuffer();
    if (computeEncoder_)
        return computeEncoder_;

    endBlitEncoding();
    @autoreleasepool
    {
        computeEncoder_ = [commandBuffer_ computeCommandEncoder];
        CV_CheckTrue(computeEncoder_ != nil, "Failed to create Metal compute encoder");
#if !__has_feature(objc_arc)
        [computeEncoder_ retain];
#endif
    }
    return computeEncoder_;
}

id<MTLBlitCommandEncoder> Context::blitEncoder()
{
    ensureCommandBuffer();
    if (blitEncoder_)
        return blitEncoder_;

    endEncoding();
    @autoreleasepool
    {
        blitEncoder_ = [commandBuffer_ blitCommandEncoder];
        CV_CheckTrue(blitEncoder_ != nil, "Failed to create Metal blit encoder");
#if !__has_feature(objc_arc)
        [blitEncoder_ retain];
#endif
    }
    return blitEncoder_;
}

id<MTLBuffer> Context::use(const std::shared_ptr<Buffer>& buffer)
{
    CV_CheckTrue(buffer != nullptr, "Metal buffer must not be null");
    recordBuffer(buffer);
    return buffer->native();
}

void Context::upload(const std::shared_ptr<Buffer>& buffer,
                     const void* data, size_t size, size_t offset)
{
    CV_CheckTrue(buffer != nullptr, "Metal upload destination must not be null");
    CV_CheckTrue(data != nullptr, "Metal upload source must not be null");
    CV_CheckLE(offset, buffer->size(), "Metal upload offset exceeds capacity");
    CV_CheckLE(size, buffer->size() - offset, "Metal upload range exceeds capacity");

    @autoreleasepool
    {
        std::shared_ptr<Buffer> staging = Buffer::createWithData(data, size);
        id<MTLBlitCommandEncoder> encoder = blitEncoder();
        [encoder copyFromBuffer:use(staging)
                   sourceOffset:0
                       toBuffer:use(buffer)
              destinationOffset:offset
                           size:size];
        didDispatch();
    }
}

void Context::download(const std::shared_ptr<Buffer>& buffer,
                       void* data, size_t size, size_t offset)
{
    CV_CheckTrue(buffer != nullptr, "Metal download source must not be null");
    CV_CheckTrue(data != nullptr, "Metal download destination must not be null");
    CV_CheckLE(offset, buffer->size(), "Metal download offset exceeds capacity");
    CV_CheckLE(size, buffer->size() - offset, "Metal download range exceeds capacity");

    @autoreleasepool
    {
        std::shared_ptr<Buffer> staging = Buffer::create(size);
        id<MTLBlitCommandEncoder> encoder = blitEncoder();
        [encoder copyFromBuffer:use(buffer)
                   sourceOffset:offset
                       toBuffer:use(staging)
              destinationOffset:0
                           size:size];
        didDispatch();
        synchronize();
        staging->download(data, size);
    }
}

MTLSize Context::threadsPerThreadgroup1D(id<MTLComputePipelineState> pipeline,
                                         size_t threadCount) const
{
    const size_t width = std::min(
        static_cast<size_t>([pipeline threadExecutionWidth]),
        static_cast<size_t>([pipeline maxTotalThreadsPerThreadgroup]));
    return MTLSizeMake(std::min(width, threadCount), 1, 1);
}

void Context::didDispatch()
{
    ++commandBufferDispatchCount_;
    if (needsCommit())
        commit();
}

bool Context::needsCommit() const
{
    return commandBufferDispatchCount_ >= MAX_OPERATIONS_PER_COMMAND_BUFFER ||
           commandBufferResourceBytes_ >= MAX_RESOURCE_BYTES_PER_COMMAND_BUFFER;
}

void Context::recordBuffer(const std::shared_ptr<Buffer>& buffer)
{
    void* resource = static_cast<void*>(buffer->native());
    if (!commandBufferResources_.insert(resource).second)
        return;

    const size_t size = buffer->size();
    if (size > std::numeric_limits<size_t>::max() - commandBufferResourceBytes_)
        commandBufferResourceBytes_ = std::numeric_limits<size_t>::max();
    else
        commandBufferResourceBytes_ += size;
}

void Context::ensureCommandBuffer()
{
    if (commandBuffer_)
        return;

    @autoreleasepool
    {
        commandBuffer_ = [commandQueue_ commandBuffer];
        CV_CheckTrue(commandBuffer_ != nil, "Failed to create Metal command buffer");
#if !__has_feature(objc_arc)
        [commandBuffer_ retain];
#endif
    }
}

void Context::endEncoding()
{
    if (!computeEncoder_)
        return;

    [computeEncoder_ endEncoding];
#if !__has_feature(objc_arc)
    [computeEncoder_ release];
#endif
    computeEncoder_ = nil;
}

void Context::endBlitEncoding()
{
    if (!blitEncoder_)
        return;

    [blitEncoder_ endEncoding];
#if !__has_feature(objc_arc)
    [blitEncoder_ release];
#endif
    blitEncoder_ = nil;
}

void Context::commit()
{
    if (!commandBuffer_)
        return;

    endEncoding();
    endBlitEncoding();
    [commandBuffer_ addCompletedHandler:^(id<MTLCommandBuffer> commandBuffer) {
        @autoreleasepool
        {
            if ([commandBuffer status] != MTLCommandBufferStatusError)
                return;

            NSError* error = [commandBuffer error];
            const char* message = error ? [[error localizedDescription] UTF8String]
                                        : "unknown error";
            CV_LOG_ERROR(NULL, "Metal command buffer failed: " << message);
        }
    }];
    [commandBuffer_ commit];
#if !__has_feature(objc_arc)
    [lastCommittedCommandBuffer_ release];
#endif
    lastCommittedCommandBuffer_ = commandBuffer_;
    commandBuffer_ = nil;
    commandBufferDispatchCount_ = 0;
    commandBufferResourceBytes_ = 0;
    commandBufferResources_.clear();
}

void Context::synchronize()
{
    @autoreleasepool
    {
        commit();
        if (lastCommittedCommandBuffer_)
        {
            [lastCommittedCommandBuffer_ waitUntilCompleted];
#if !__has_feature(objc_arc)
            [lastCommittedCommandBuffer_ release];
#endif
            lastCommittedCommandBuffer_ = nil;
        }
    }
}

Context& Context::get()
{
    static thread_local std::unique_ptr<Context> context;
    if (!context)
        context.reset(new Context(Device::instance()));
    return *context;
}

}}}  // namespace cv::dnn::metal
