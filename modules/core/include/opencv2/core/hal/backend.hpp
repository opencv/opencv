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

/** @brief Operation identifiers passed to Backend::support() and Backend::run().
 *
 * Values 100–199 are reserved for built-in image-processing operations.
 * Third-party backends may define additional IDs starting at 1000.
 */
enum GpuOpId
{
    GPU_OP_RESIZE           = 100,
    GPU_OP_GAUSSIAN_BLUR    = 101,
    GPU_OP_CVT_COLOR        = 102,
    GPU_OP_THRESHOLD        = 103,
    GPU_OP_ADD              = 104,
    GPU_OP_SUBTRACT         = 105,
    GPU_OP_MULTIPLY         = 106,
    GPU_OP_WARP_AFFINE      = 107,
    GPU_OP_WARP_PERSPECTIVE = 108
};

/** @brief Abstract GPU backend interface.
 *
 * A backend encapsulates one GPU execution environment (CUDA, Vulkan, HIP, …).
 * Concrete subclasses are registered at run-time; the CV_GPU_RUN macro queries
 * the active backend and delegates execution without any CUDA/Vulkan/HIP
 * dependency in the caller.
 *
 * Implementors must provide:
 *   - support()  — advertise which operations this backend handles
 *   - run()      — execute one operation
 *
 * Optionally override allocator() to supply a custom MatAllocator so that
 * intermediate UMat buffers stay on the device between operations.
 */
class CV_EXPORTS Backend
{
public:
    virtual ~Backend() {}

    /** @brief Returns true if this backend can handle the given operation.
     *
     * @param op_id  One of the GpuOpId constants (or a user-defined extension).
     */
    virtual bool support(int op_id) const = 0;

    /** @brief Execute an operation on the backend.
     *
     * @param op_id   Operation identifier; support(op_id) must be true.
     * @param src     Source array.
     * @param dst     Destination array.
     * @param param1  Integer parameter 1 (semantics are op-specific).
     * @param param2  Integer parameter 2 (semantics are op-specific).
     * @param fparam1 Floating-point parameter 1 (semantics are op-specific).
     * @param fparam2 Floating-point parameter 2 (semantics are op-specific).
     * @return true on success; false signals the caller to fall back to CPU.
     */
    virtual bool run(int op_id,
                     InputArray  src,
                     OutputArray dst,
                     int    param1  = 0,
                     int    param2  = 0,
                     double fparam1 = 0.0,
                     double fparam2 = 0.0) = 0;

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
