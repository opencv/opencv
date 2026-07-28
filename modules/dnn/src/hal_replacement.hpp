// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_HAL_REPLACEMENT_HPP
#define OPENCV_DNN_HAL_REPLACEMENT_HPP

#include "opencv2/core/hal/interface.h"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100)
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

//! @addtogroup dnn
//! @{
//!
//! HAL replacement hooks for the compute kernels of the DNN engine.
//!
//! An accelerated backend (e.g. the RISC-V RVV HAL) overrides a kernel by
//! redefining the matching @c cv_hal_dnn_* macro to point at its own function;
//! the default @c hal_ni_* below returns @c CV_HAL_ERROR_NOT_IMPLEMENTED, so the
//! built-in implementation runs unchanged when no backend is present.
//!
//! Tensors use the engine's blocked NCDHWc layout described by @c cv::dnn::ConvState:
//! @c N * C1 channel-blocks of @c C0 channels each, laid out as [N, C1, ...spatial..., C0].
//!
//! **Parallelism is owned by OpenCV, not by the HAL.** Each hook receives a task
//! range @c [task_start, task_end) over the @c [0, N*C1) block-plane index and must
//! compute *only* that slice of the output tensor. The engine drives @c parallel_for_
//! and calls the hook once per worker range, so a hook must never spawn its own threads.

// The pooling and depthwise-convolution geometry crosses the boundary as a flat, stable C
// argument list (no dnn types): @c C0 = channel block; @c insize / @c outsize = the [3]
// input/output spatial dims in a fixed Z,Y,X frame (unused leading dims = 1); @c strides [3];
// @c pads [6] (begin[0..2] + end[3..5]); @c inner [6] = the padding-free interior bounds;
// @c coordtab [ksize*3] = per-tap (dz,dy,dx); @c ofstab [ksize] = per-tap flat input offset
// (interior).

/** @brief Max pooling over a slice [task_start, task_end) of the output (blocked NCDHWc, CV_32F). */
inline int hal_ni_dnn_maxpool3d32f(const float* inp_data, float* out_data, int C0,
                                 const int* insize, const int* outsize, const int* strides,
                                 const int* pads, const int* inner, const int* coordtab,
                                 const int* ofstab, int ksize, int task_start, int task_end)
{ return CV_HAL_ERROR_NOT_IMPLEMENTED; }

/** @brief Average pooling over a slice [task_start, task_end) of the output (blocked NCDHWc,
CV_32F); a nonzero @c count_include_pad divides by the full kernel size (counts padded cells). */
inline int hal_ni_dnn_avgpool3d32f(const float* inp_data, float* out_data, int C0,
                                 const int* insize, const int* outsize, const int* strides,
                                 const int* pads, const int* inner, const int* coordtab,
                                 const int* ofstab, int ksize, int count_include_pad,
                                 int task_start, int task_end)
{ return CV_HAL_ERROR_NOT_IMPLEMENTED; }

//! @brief Depthwise convolution over a slice [task_start, task_end) of the output
//! (blocked NCDHWc, CV_32F), with the fused post-op @c out = act(in*W + bias, scaled).
//!
//! In addition to the geometry above, this carries the per-block weights and the fused
//! epilogue. @c weights is the repacked @c C1*ksize*C0 tensor (block @c b at @c b*ksize*C0);
//! @c scale / @c bias are optional per-channel vectors of length @c C (null => 1 / 0). The
//! task index runs over @c [0, N*C1) block-planes decomposed as @c n=nc1/C1,
//! @c c_base=(nc1-n*C1)*C0. @c residual (optional, null when absent) is added before the
//! activation. The activation is passed enum-free: @c out = min(s>=0 ? s : s*alpha, maxval),
//! where @c alpha is @c prelu_slope[c] when @c prelu_slope != null else @c default_alpha
//! (1 => identity, 0 => ReLU, in (0,1) => leaky), and @c maxval clamps (FLT_MAX => none).
//! A generic (function-pointer) activation the engine cannot express here is applied by the
//! engine after this hook returns.
inline int hal_ni_dnn_depthwise_conv32f(const float* inp_data, const float* residual_data,
                                        float* out_data, const float* weights,
                                        const float* scale, const float* bias,
                                        int C, int C0, int C1,
                                        const int* insize, const int* outsize, const int* strides,
                                        const int* pads, const int* inner, const int* coordtab,
                                        const int* ofstab, int ksize,
                                        float maxval, float default_alpha, const float* prelu_slope,
                                        int task_start, int task_end)
{ return CV_HAL_ERROR_NOT_IMPLEMENTED; }

//! @cond IGNORED
#define cv_hal_dnn_maxpool3d32f hal_ni_dnn_maxpool3d32f
#define cv_hal_dnn_avgpool3d32f hal_ni_dnn_avgpool3d32f
#define cv_hal_dnn_depthwise_conv32f hal_ni_dnn_depthwise_conv32f
//! @endcond

//! @}

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

// Pull in the registered HAL's overrides (e.g. the RISC-V RVV HAL), which
// #undef/#redefine the cv_hal_dnn_* macros above to point at their own kernels.
// The file is always generated (empty when no HAL is registered), so this is a
// no-op on default builds.
#include "custom_hal.hpp"

//! @cond IGNORED
// Call an overridable DNN HAL hook; on CV_HAL_ERROR_OK return from the current
// scope (skipping the built-in fallback), on NOT_IMPLEMENTED fall through to it.
// Used inside a parallel_for_ worker lambda, so the early return leaves that one
// task range to the HAL while the rest of the engine is unaffected.
#define CALL_HAL(name, fun, ...) \
{ \
    int res = __CV_EXPAND(fun(__VA_ARGS__)); \
    if (res == CV_HAL_ERROR_OK) \
        return; \
    else if (res != CV_HAL_ERROR_NOT_IMPLEMENTED) \
        CV_Error_(cv::Error::StsInternal, \
            ("HAL implementation " CVAUX_STR(name) " ==> " CVAUX_STR(fun) " returned %d (0x%08x)", res, res)); \
}
//! @endcond

#endif // OPENCV_DNN_HAL_REPLACEMENT_HPP
