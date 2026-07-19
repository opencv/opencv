#include "../../precomp.hpp"
#include "allocator.hpp"

#include "device.hpp"

namespace cv { namespace dnn { namespace metal {

static const MTLResourceOptions BUFFER_OPTIONS =
    MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked;

Allocator::Allocator()
    : device_(Device::instance())
{
    CV_CheckTrue(device_ != nullptr, "Metal device is unavailable");

    @autoreleasepool
    {
        MTLHeapDescriptor* descriptor = [[MTLHeapDescriptor alloc] init];
        [descriptor setSize:HEAP_SIZE];
        [descriptor setResourceOptions:BUFFER_OPTIONS];
        heap_ = [device_->native() newHeapWithDescriptor:descriptor];
#if !__has_feature(objc_arc)
        [descriptor release];
#endif
    }
}

Allocator::~Allocator()
{
    std::lock_guard<std::mutex> lock(mutex_);
#if !__has_feature(objc_arc)
    [heap_ release];
#endif
    heap_ = nil;
}

Allocator& Allocator::get()
{
    static Allocator allocator;
    return allocator;
}

id<MTLBuffer> Allocator::allocate(size_t size)
{
    CV_CheckGT(size, static_cast<size_t>(0), "Metal buffer size must be positive");
    CV_CheckLE(size, static_cast<size_t>([device_->native() maxBufferLength]),
               "Metal buffer size exceeds the device limit");

    std::lock_guard<std::mutex> lock(mutex_);

    id<MTLBuffer> buffer = nil;
    if (size < SMALL_BUFFER_SIZE && heap_ != nil)
        buffer = [heap_ newBufferWithLength:size options:BUFFER_OPTIONS];
    if (buffer == nil)
        buffer = [device_->native() newBufferWithLength:size options:BUFFER_OPTIONS];

    CV_CheckTrue(buffer != nil, "Failed to allocate Metal buffer");
    return buffer;
}

void Allocator::free(id<MTLBuffer>& buffer)
{
    if (buffer == nil)
        return;

    std::lock_guard<std::mutex> lock(mutex_);
#if !__has_feature(objc_arc)
    [buffer release];
#endif
    buffer = nil;
}

}}} // namespace cv::dnn::metal
