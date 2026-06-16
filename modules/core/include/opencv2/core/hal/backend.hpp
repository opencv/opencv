// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CORE_HAL_BACKEND_HPP
#define OPENCV_CORE_HAL_BACKEND_HPP

#include "opencv2/core/cvdef.h"
#include "opencv2/core/mat.hpp"

namespace cv { namespace hal {

//! @addtogroup core_hal_backend
//! @{

/** @brief Abstract GPU backend interface.
 *
 * A backend encapsulates one GPU execution environment (CUDA, Vulkan, HIP, …).
 * Concrete subclasses are registered at run-time; the CV_GPU_RUN macro fetches
 * the backend attached to the source UMat and calls the matching operation,
 * with no CUDA/Vulkan/HIP dependency in the caller.
 *
 * Each operation is a typed virtual method (mirrors the CPU HAL's one-function-
 * per-op style). A backend overrides only the operations it implements; the
 * default implementations return false, which makes CV_GPU_RUN fall through to
 * the existing CPU/OpenCL path. There is no operation-id enum and no generic
 * run(): the method name *is* the operation selector.
 *
 * Optionally override allocator() to supply a custom MatAllocator so that
 * intermediate UMat buffers stay on the device between operations.
 *
 * Each method returns true if it executed the operation (caller returns),
 * false to fall back to CPU.
 */
class CV_EXPORTS Backend
{
public:
    virtual ~Backend() {}

    //! resize: dsize, inv_scale_x, inv_scale_y, interpolation
    virtual bool resize(InputArray, OutputArray, Size, double, double, int) { return false; }

    //! Gaussian blur: ksize, sigma1, sigma2
    virtual bool gaussianBlur(InputArray, OutputArray, Size, double, double) { return false; }

    //! color conversion: code, dst channel count (dcn)
    virtual bool cvtColor(InputArray, OutputArray, int, int) { return false; }

    //! threshold: thresh, maxval, type
    virtual bool threshold(InputArray, OutputArray, double, double, int) { return false; }

    /** @brief Return a device-aware MatAllocator, or nullptr to use the default.
     *
     * When non-null the allocator is used for UMat buffers passed through this
     * backend, keeping data on the device between operations.
     */
    virtual MatAllocator* allocator() const { return NULL; }
};

//! @}

} // namespace hal
} // namespace cv

#endif // OPENCV_CORE_HAL_BACKEND_HPP
