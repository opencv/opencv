#ifndef OPENCV_DNN_METAL_RUNTIME_DEVICE_HPP
#define OPENCV_DNN_METAL_RUNTIME_DEVICE_HPP

#import <Metal/Metal.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace cv { namespace dnn { namespace metal {

class Device
{
public:
    static std::shared_ptr<Device> instance();

    ~Device();

    id<MTLDevice> native() const;
    std::string name() const;
    bool libraryAvailable() const;
    bool hasUnifiedMemory() const;
    id<MTLComputePipelineState> pipelineState(const std::string& kernelName);

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&&) = delete;
    Device& operator=(Device&&) = delete;

private:
    explicit Device(id<MTLDevice> device);

    id<MTLDevice> device_ = nil;
    id<MTLLibrary> library_ = nil;
    std::string libraryError_;
    std::mutex pipelineMutex_;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelineStates_;
};

}}}  // namespace cv::dnn::metal

#endif  // OPENCV_DNN_METAL_RUNTIME_DEVICE_HPP
