#include "../precomp.hpp"
#include "metal.hpp"

#include "runtime/context.hpp"
#include "runtime/device.hpp"

#include <opencv2/core/utils/logger.hpp>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace cv { namespace dnn { namespace metal {

bool isAvailable() noexcept
{
    try
    {
        return Device::instance() != nullptr && Device::instance()->libraryAvailable();
    }
    catch (...)
    {
        return false;
    }
}

bool startCapture(const std::string& path)
{
    @autoreleasepool
    {
        Context::get().synchronize();

        const std::shared_ptr<Device> device = Device::instance();
        CV_CheckTrue(device != nullptr, "Metal device is unavailable");

        MTLCaptureManager* manager = [MTLCaptureManager sharedCaptureManager];
        if (![manager supportsDestination:MTLCaptureDestinationGPUTraceDocument])
        {
            CV_LOG_WARNING(NULL, "Metal GPU trace capture is not supported");
            return false;
        }

        MTLCaptureDescriptor* descriptor = [[MTLCaptureDescriptor alloc] init];
        descriptor.captureObject = device->native();
        descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
        descriptor.outputURL = [NSURL fileURLWithPath:
            [NSString stringWithUTF8String:path.c_str()]];

        NSError* error = nil;
        const BOOL started = [manager startCaptureWithDescriptor:descriptor error:&error];
#if !__has_feature(objc_arc)
        [descriptor release];
#endif
        if (!started)
        {
            const char* message = error ? [[error localizedDescription] UTF8String]
                                        : "unknown error";
            CV_LOG_WARNING(NULL, "Failed to start Metal capture: " << message);
            return false;
        }
        return true;
    }
}

void stopCapture()
{
    Context::get().synchronize();
    [[MTLCaptureManager sharedCaptureManager] stopCapture];
}

}}} // namespace cv::dnn::metal
