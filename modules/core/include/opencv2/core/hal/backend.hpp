/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) ARISING IN ANY WAY OUT OF
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

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
