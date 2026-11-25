// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_NON_CPU_HAL_GPU_HAL_IMGPROC_HPP
#define OPENCV_NON_CPU_HAL_GPU_HAL_IMGPROC_HPP

// Non-CPU HAL, GPU: the imgproc kernels and operations, each family in src/<family>.impl.hpp,
// on the runtime of core's src/accel/gpu/gpu_runtime.hpp, which is included first. Images are of
// @p depth (CV_8U, ...) with @p cn channels; borders are cv::BorderTypes.

#include <stddef.h>

namespace cv { namespace accel { namespace CV_GPU_NS { namespace kernels {

//! cv::InterpolationFlags, which core does not see
enum { INTER_NEAREST = 0, INTER_LINEAR = 1 };

//! the longest GaussianBlur kernel, in taps
const int GAUSSIAN_MAX_TAPS = 63;
/** @brief GaussianBlur, a row pass into @p buf, then a column pass. @p kx, @p ky and @p buf are
CV_16U for CV_8U images, CV_32U for CV_16U (fixed point, as the CPU's), CV_32F otherwise. */
CV_GPU(Error_t) gaussianBlur(const unsigned char* src, size_t sstep, unsigned char* buf, size_t bstep,
                             unsigned char* dst, size_t dstep, int rows, int cols, int depth, int cn,
                             const void* kx, int kxn, const void* ky, int kyn, int border, CV_GPU(Stream_t) s);
/** @brief warpAffine with the inverse map @p M (2x3), INTER_NEAREST or INTER_LINEAR, @p bval the
BORDER_CONSTANT pixel already in the range of @p depth. */
CV_GPU(Error_t) warpAffine(const unsigned char* src, size_t sstep, int srows, int scols,
                           unsigned char* dst, size_t dstep, int drows, int dcols, int depth, int cn,
                           const float* M, int interpolation, int border, const float* bval,
                           CV_GPU(Stream_t) s);

}}}} // namespace cv::accel::CV_GPU_NS::kernels

#ifndef CV_ACCEL_GPU_KERNELS

namespace cv { namespace accel { namespace CV_GPU_NS {

namespace {

//! the imgproc operations; each is defined in the file of its family
class GpuImgprocOps CV_FINAL : public ImgprocOps
{
public:
    bool gaussianBlur(const UMat& src, UMat& dst, const Mat& kx, const Mat& ky, int borderType,
                      Stream& stream) CV_OVERRIDE;    // smooth.impl.hpp
    bool warpAffine(const UMat& src, UMat& dst, InputArray M, int flags, int borderType,
                    const Scalar& borderValue, Stream& stream) CV_OVERRIDE;    // warp.impl.hpp
};

Ptr<ImgprocOps> createImgprocOps()
{
    return makePtr<GpuImgprocOps>();
}

} // namespace

}}} // namespace cv::accel::CV_GPU_NS

#endif // CV_ACCEL_GPU_KERNELS

#endif // OPENCV_NON_CPU_HAL_GPU_HAL_IMGPROC_HPP
