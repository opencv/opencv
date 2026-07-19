#ifndef OPENCV_DNN_METAL_RUNTIME_CONTEXT_HPP
#define OPENCV_DNN_METAL_RUNTIME_CONTEXT_HPP

#import <Metal/Metal.h>

#include <memory>
#include <unordered_set>

namespace cv { namespace dnn { namespace metal {

class Buffer;
class Device;

// Holds mutable Metal execution state for the calling thread.  It is shared by
// all DNN Nets executed on that thread and has no ownership relationship with
// any Net, Layer or Tensor.
class Context
{
public:
    static Context& get();

    ~Context();

    // The returned encoder is owned by this context and may be ended by
    // didDispatch() when the current command buffer reaches its limits.
    // Operations must reacquire it before encoding each dispatch.
    id<MTLComputeCommandEncoder> computeEncoder();
    id<MTLBuffer> use(const std::shared_ptr<Buffer>& buffer);
    void upload(const std::shared_ptr<Buffer>& buffer,
                const void* data, size_t size, size_t offset = 0);
    void download(const std::shared_ptr<Buffer>& buffer,
                  void* data, size_t size, size_t offset = 0);
    MTLSize threadsPerThreadgroup1D(id<MTLComputePipelineState> pipeline,
                                    size_t threadCount) const;
    void didDispatch();
    void commit();
    void synchronize();

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) = delete;
    Context& operator=(Context&&) = delete;

private:
    explicit Context(const std::shared_ptr<Device>& device);

    void ensureCommandBuffer();
    id<MTLBlitCommandEncoder> blitEncoder();
    void endEncoding();
    void endBlitEncoding();
    bool needsCommit() const;
    void recordBuffer(const std::shared_ptr<Buffer>& buffer);

    std::shared_ptr<Device> device_;
    id<MTLCommandQueue> commandQueue_ = nil;
    id<MTLCommandBuffer> commandBuffer_ = nil;
    id<MTLComputeCommandEncoder> computeEncoder_ = nil;
    id<MTLBlitCommandEncoder> blitEncoder_ = nil;
    id<MTLCommandBuffer> lastCommittedCommandBuffer_ = nil;
    size_t commandBufferDispatchCount_ = 0;
    size_t commandBufferResourceBytes_ = 0;
    std::unordered_set<void*> commandBufferResources_;
};

}}}  // namespace cv::dnn::metal

#endif  // OPENCV_DNN_METAL_RUNTIME_CONTEXT_HPP
