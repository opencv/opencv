// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Non-CPU HAL: the HIP backend (AMD GPUs); all but the choice of memory is shared with CUDA.
//
// Discrete GPUs use device memory from a stream-ordered pool; APUs use pinned host memory mapped
// into the GPU, which the CPU uses without copies.

#define CV_ACCEL_GPU_HIP
#include "accel_gpu.hpp"
#include "backend.impl.hpp"

namespace cv { namespace accel { namespace hip_impl {

namespace {

class HipBackend CV_FINAL : public GpuBackend
{
public:
    HipBackend()
    {
        for (size_t i = 0; i < devs_.size(); i++)
        {
            GpuDevice& d = *devs_[i];
            d.shared = d.prop.integrated && d.prop.canMapHostMemory;
        }
    }

    String name() const CV_OVERRIDE { return "hip"; }
};

} // namespace

Ptr<Backend> createHipBackend()
{
    return makePtr<HipBackend>();
}

}}} // namespace cv::accel::hip_impl
