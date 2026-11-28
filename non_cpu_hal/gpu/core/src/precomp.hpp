// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_NON_CPU_HAL_GPU_CORE_PRECOMP_HPP
#define OPENCV_NON_CPU_HAL_GPU_CORE_PRECOMP_HPP

// Non-CPU HAL, GPU, core: what the families share, then the families, one file each (included by
// ../../precomp.hpp). The kernels move elements of @p esz bytes as opaque groups of bytes, so one kernel
// serves every type of the same element size; rows are @p sstep / @p dstep / @p step bytes apart.

#ifdef CV_ACCEL_GPU_KERNELS

namespace cv { namespace accel { namespace CV_GPU_NS { namespace kernels {

namespace {

//! the largest power of two dividing @p n, up to 16
constexpr int elemAlign(int n)
{
    return n % 16 == 0 ? 16 : n % 8 == 0 ? 8 : n % 4 == 0 ? 4 : n % 2 == 0 ? 2 : 1;
}
// one element of N bytes, aligned as far as N allows, so a copy is one wide load and store
template<int N> struct alignas(elemAlign(N)) Elem { unsigned char b[N]; };
// the same bytes at any alignment, for rows whose start or step is not a multiple of that
template<int N> struct Bytes { unsigned char b[N]; };

inline bool aligned(const void* src, size_t sstep, const void* dst, size_t dstep, size_t a)
{
    return (((size_t)src | sstep | (size_t)dst | dstep) & (a - 1)) == 0;
}

} // namespace

//! returns fn<esz>(...) for the element sizes supportedElementSize() accepts
#define CV_ACCEL_GPU_BY_SIZE(fn, ...) \
    switch (esz) \
    { \
    case 1: return fn<1>(__VA_ARGS__); \
    case 2: return fn<2>(__VA_ARGS__); \
    case 3: return fn<3>(__VA_ARGS__); \
    case 4: return fn<4>(__VA_ARGS__); \
    case 6: return fn<6>(__VA_ARGS__); \
    case 8: return fn<8>(__VA_ARGS__); \
    case 12: return fn<12>(__VA_ARGS__); \
    case 16: return fn<16>(__VA_ARGS__); \
    case 24: return fn<24>(__VA_ARGS__); \
    case 32: return fn<32>(__VA_ARGS__); \
    default: return CV_GPU(ErrorInvalidValue); \
    }

}}}} // namespace cv::accel::CV_GPU_NS::kernels

#else // the operations

#include "gpu/core/include/gpu_hal_core.hpp"

namespace cv { namespace accel { namespace CV_GPU_NS {

namespace kernels {

//! the element sizes the kernels move (CV_ACCEL_GPU_BY_SIZE)
inline bool supportedElementSize(int esz)
{
    switch (esz)
    {
    case 1: case 2: case 3: case 4: case 6: case 8: case 12: case 16: case 24: case 32:
        return true;
    default:
        return false;
    }
}

} // namespace kernels

namespace {

/* Whether the kernels can run on @p src and @p dst: 2-D, an element size they move, no Mat view
   alive (UMat::handle() needs that), and either separate buffers or the same view (in place).
   Other views of one buffer may overlap, which the kernels do not handle. */
bool runnable(const UMat& src, const UMat& dst)
{
    return src.dims == 2 && kernels::supportedElementSize((int)src.elemSize()) &&
           src.u->refcount == 0 && dst.u->refcount == 0 &&
           (src.u != dst.u || (src.offset == dst.offset && src.step[0] == dst.step[0]));
}

} // namespace

}}} // namespace cv::accel::CV_GPU_NS

#endif // CV_ACCEL_GPU_KERNELS

#include "flip.impl.hpp"
#include "transpose.impl.hpp"

#ifdef CV_ACCEL_GPU_KERNELS

#undef CV_ACCEL_GPU_BY_SIZE

#else

namespace cv { namespace accel { namespace CV_GPU_NS {

namespace {

//! the core operations: one mixin per family
class GpuCoreOps CV_FINAL : public GpuTransposeOps<GpuFlipOps<CoreOps> > {};

Ptr<CoreOps> createCoreOps()
{
    return makePtr<GpuCoreOps>();
}

} // namespace

}}} // namespace cv::accel::CV_GPU_NS

#endif // CV_ACCEL_GPU_KERNELS

#endif // OPENCV_NON_CPU_HAL_GPU_CORE_PRECOMP_HPP
