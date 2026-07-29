#include "../../precomp.hpp"
#include "device.hpp"

#include <opencv2/core/utils/logger.hpp>

#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>

extern "C" {
extern const unsigned char opencv_dnn_metallib_start[];
extern const unsigned char opencv_dnn_metallib_end[];
}

namespace cv { namespace dnn { namespace metal {

static id<MTLLibrary> loadEmbeddedLibrary(id<MTLDevice> device,
                                          std::string& errorMessage)
{
    const void* bytes = opencv_dnn_metallib_start;
    const size_t size = static_cast<size_t>(opencv_dnn_metallib_end -
                                            opencv_dnn_metallib_start);
    if (size == 0)
    {
        errorMessage = "Embedded Metal library is empty";
        return nil;
    }

    @autoreleasepool
    {
        dispatch_data_t data = dispatch_data_create(
            bytes, size, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^{});
        if (!data)
        {
            errorMessage = "Failed to create dispatch data for the embedded Metal library";
            return nil;
        }

        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithData:data error:&error];
#if OS_OBJECT_USE_OBJC
#if !__has_feature(objc_arc)
        [(id)data release];
#endif
#else
        dispatch_release(data);
#endif
        if (!library)
        {
            errorMessage = error ? std::string([[error localizedDescription] UTF8String])
                                 : std::string("unknown error");
        }
        return library;
    }
}

Device::Device(id<MTLDevice> device) : device_(device)
{
    CV_CheckTrue(device_ != nil, "Metal device must not be null");
    library_ = loadEmbeddedLibrary(device_, libraryError_);
}

Device::~Device()
{
#if !__has_feature(objc_arc)
    for (const auto& entry : pipelineStates_)
        [entry.second release];
    [library_ release];
    [device_ release];
#endif
    pipelineStates_.clear();
}

std::shared_ptr<Device> Device::instance()
{
    static const std::shared_ptr<Device> d = []() -> std::shared_ptr<Device>
    {
        @autoreleasepool
        {
            id<MTLDevice> nativeDevice = MTLCreateSystemDefaultDevice();
            if (!nativeDevice)
                return std::shared_ptr<Device>();
            std::shared_ptr<Device> result(new Device(nativeDevice));
            if (!result->library_)
            {
                CV_Error_(Error::GpuApiCallError,
                          ("Failed to initialize Metal library: %s",
                           result->libraryError_.c_str()));
            }
            return result;
        }
    }();
    return d;
}

id<MTLDevice> Device::native() const
{
    return device_;
}

std::string Device::name() const
{
    return device_ ? std::string([[device_ name] UTF8String]) : std::string();
}

bool Device::libraryAvailable() const
{
    return library_ != nil;
}

bool Device::hasUnifiedMemory() const
{
    if (!device_)
        return false;
    return [device_ hasUnifiedMemory];
}

id<MTLComputePipelineState> Device::pipelineState(const std::string& kernelName)
{
    std::lock_guard<std::mutex> lock(pipelineMutex_);
    std::unordered_map<std::string, id<MTLComputePipelineState>>::const_iterator found =
        pipelineStates_.find(kernelName);
    if (found != pipelineStates_.end())
        return found->second;

    @autoreleasepool
    {
        NSString* name = [NSString stringWithUTF8String:kernelName.c_str()];
        id<MTLFunction> function = [library_ newFunctionWithName:name];
        if (!function)
            CV_Error_(Error::StsObjectNotFound,
                      ("Metal kernel '%s' was not found", kernelName.c_str()));

        NSError* error = nil;
        id<MTLComputePipelineState> nativePipeline =
            [device_ newComputePipelineStateWithFunction:function error:&error];
#if !__has_feature(objc_arc)
        [function release];
#endif
        if (!nativePipeline)
        {
            const char* message = error ? [[error localizedDescription] UTF8String] : "unknown error";
            CV_Error_(Error::GpuApiCallError,
                      ("Failed to create Metal pipeline '%s': %s", kernelName.c_str(), message));
        }

        pipelineStates_[kernelName] = nativePipeline;
        return nativePipeline;
    }
}

}}} // namespace cv::dnn::metal
