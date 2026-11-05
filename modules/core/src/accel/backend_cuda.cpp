// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Non-CPU HAL: the CUDA backend; all but the choice of memory lives in gpu/.
//
// Discrete GPUs use device memory from a stream-ordered pool with pinned host copies; integrated
// GPUs from Xavier on use pinned memory mapped into the GPU, so nothing is copied.

#include "../precomp.hpp"
#include "accel_impl.hpp"

#ifdef HAVE_CUDA

#include "gpu/gpu_backend.impl.hpp"

namespace cv { namespace accel { namespace cuda_impl {

namespace {

class CudaBackend CV_FINAL : public GpuBackend
{
public:
    CudaBackend()
    {
        for (size_t i = 0; i < devs_.size(); i++)
        {
            GpuDevice& d = *devs_[i];
            const int cc = d.prop.major * 10 + d.prop.minor;
            // before Xavier, pinned memory is uncached for the CPU: copies are faster there
            d.shared = d.prop.integrated && d.prop.canMapHostMemory && cc >= 72;
            // one kind per device: device memory, or shared memory on an integrated part (Jetson)
            d.kinds = d.shared ? MEMORY_SHARED_COHERENT : MEMORY_DEVICE;
        }
    }

    String name() const CV_OVERRIDE { return "cuda"; }
};

} // namespace

Ptr<Backend> createCudaBackend()
{
    return makePtr<CudaBackend>();
}

}}} // namespace cv::accel::cuda_impl

#endif // HAVE_CUDA
