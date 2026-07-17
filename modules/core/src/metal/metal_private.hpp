// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_MISC_METAL_PRIVATE_HPP
#define OPENCV_CORE_MISC_METAL_PRIVATE_HPP

#include "../precomp.hpp"

#ifdef HAVE_METAL

#include "../metal.hpp"
#include "../umatrix.hpp"
#include "opencv2/core/metal.hpp"
#include "opencv2/core/bufferpool.hpp"

#import <Metal/Metal.h>

namespace cv {
namespace metal {

enum MetalStorageKind
{
    METAL_STORAGE_SHARED,
    METAL_STORAGE_MANAGED,
    METAL_STORAGE_PRIVATE
};

struct MetalBuffer
{
    MetalBuffer(id<MTLBuffer> buffer_, size_t size_, MetalStorageKind storageKind_, MTLResourceOptions storageOptions_);
    ~MetalBuffer();

    id<MTLBuffer> buffer;
    size_t size;
    MetalStorageKind storageKind;
    bool hostVisible;
    MTLResourceOptions storageOptions;
};

class MetalContext
{
public:
    MetalContext();
    ~MetalContext();

    bool valid() const;
    id<MTLDevice> device() const;
    id<MTLCommandQueue> queue() const;

private:
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
};

std::shared_ptr<MetalContext> getMetalContext();
MetalBuffer* getBuffer(UMatData* u);
uchar* getContents(UMatData* u);
MTLResourceOptions getStorageOptions(const std::shared_ptr<MetalContext>& ctx, UMatUsageFlags usageFlags, MetalStorageKind& storageKind);
MetalBuffer* allocateBuffer(const std::shared_ptr<MetalContext>& ctx, size_t size, MTLResourceOptions storageOptions, MetalStorageKind storageKind);
void releaseBuffer(MetalBuffer* buffer);
BufferPoolController* getMetalBufferPoolController();
bool isHostVisible(UMatData* u);
bool isManagedStorage(UMatData* u);
void synchronizeForCpuRead(UMatData* u);
void notifyCpuWrite(UMatData* u);
id<MTLBuffer> newSharedBuffer(const std::shared_ptr<MetalContext>& ctx, size_t size);
void blitCopy(const std::shared_ptr<MetalContext>& ctx,
              id<MTLBuffer> src, size_t srcofs,
              id<MTLBuffer> dst, size_t dstofs,
              size_t size);
void downloadToHost(UMatData* u, void* dst);
void uploadFromHost(UMatData* u, const void* src);
bool isContinuousRegion(int dims, const size_t sz[], const size_t ofs[],
                        const size_t step[], size_t& rawofs, size_t& total);
MatAllocator* getMetalAllocator_();

} // namespace metal
} // namespace cv

#endif // HAVE_METAL
#endif // OPENCV_CORE_MISC_METAL_PRIVATE_HPP
