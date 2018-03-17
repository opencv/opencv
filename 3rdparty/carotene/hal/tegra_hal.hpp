/*
 * By downloading, copying, installing or using the software you agree to this license.
 * If you do not agree to this license, do not download, install,
 * copy or use the software.
 *
 *
 *                           License Agreement
 *                For Open Source Computer Vision Library
 *                        (3-clause BSD License)
 *
 * Copyright (C) 2016, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *   * Neither the names of the copyright holders nor the names of the contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 * This software is provided by the copyright holders and contributors "as is" and
 * any express or implied warranties, including, but not limited to, the implied
 * warranties of merchantability and fitness for a particular purpose are disclaimed.
 * In no event shall copyright holders or contributors be liable for any direct,
 * indirect, incidental, special, exemplary, or consequential damages
 * (including, but not limited to, procurement of substitute goods or services;
 * loss of use, data, or profits; or business interruption) however caused
 * and on any theory of liability, whether in contract, strict liability,
 * or tort (including negligence or otherwise) arising in any way out of
 * the use of this software, even if advised of the possibility of such damage.
 */

#ifndef _tegra_hal_H_INCLUDED_
#define _tegra_hal_H_INCLUDED_

#define CAROTENE_NS carotene_o4t

#include "carotene/functions.hpp"
#include <cstddef>
#include <cstring>
#include <vector>
#include <opencv2/core/base.hpp>

#define RANGE_DATA(type, base, step) reinterpret_cast<type*>(const_cast<char *>(reinterpret_cast<const char *>(base)) + static_cast<size_t>(range.start) * step)

#define PARALLEL_CORE 0
#if PARALLEL_CORE

#define SRC_ARG1 ST * src1_data_, size_t src1_step_,
#define SRC_STORE1 src1_data(src1_data_), src1_step(src1_step_),
#define SRC_VAR1 ST * src1_data; \
                 size_t src1_step;
#define SRC_ARG2 ST * src1_data_, size_t src1_step_, \
                 ST * src2_data_, size_t src2_step_,
#define SRC_STORE2 src1_data(src1_data_), src1_step(src1_step_), \
                   src2_data(src2_data_), src2_step(src2_step_),
#define SRC_VAR2 ST * src1_data; \
                 size_t src1_step; \
                 ST * src2_data; \
                 size_t src2_step;

#define DST_ARG1 DT * dst1_data_, size_t dst1_step_,
#define DST_STORE1 dst1_data(dst1_data_), dst1_step(dst1_step_),
#define DST_VAR1 DT * dst1_data; \
                 size_t dst1_step;

#define SCALE_ARG0
#define SCALE_STORE0
#define SCALE_VAR0
#define SCALE_ARG1 , double scale_
#define SCALE_STORE1 , scale(scale_)
#define SCALE_VAR1 double scale;
#define SCALE_ARG3 , const double *scales_
#define SCALE_STORE3 , scales(scales_, scales_ + 3)
#define SCALE_VAR3 std::vector<double> scales;

#define TegraGenOp_Invoker(name, func, src_cnt, dst_cnt, scale_cnt, ...) \
template <typename ST, typename DT> \
class TegraGenOp_##name##_Invoker : public cv::ParallelLoopBody \
{ \
public: \
    TegraGenOp_##name##_Invoker(SRC_ARG##src_cnt \
                                DST_ARG##dst_cnt \
                                int width_, int height_ \
                                SCALE_ARG##scale_cnt) : \
        cv::ParallelLoopBody(), SRC_STORE##src_cnt \
                                DST_STORE##dst_cnt \
                                width(width_), height(height_) \
                                SCALE_STORE##scale_cnt {} \
    virtual void operator()(const cv::Range& range) const \
    { \
        CAROTENE_NS::func(CAROTENE_NS::Size2D(width, range.end-range.start), __VA_ARGS__); \
    } \
private: \
    SRC_VAR##src_cnt \
    DST_VAR##dst_cnt \
    int width, height; \
    SCALE_VAR##scale_cnt \
    const TegraGenOp_##name##_Invoker& operator= (const TegraGenOp_##name##_Invoker&); \
};

#define TegraBinaryOp_Invoker(name, func) TegraGenOp_Invoker(name, func, 2, 1, 0, \
                                                             RANGE_DATA(ST, src1_data, src1_step), src1_step, \
                                                             RANGE_DATA(ST, src2_data, src2_step), src2_step, \
                                                             RANGE_DATA(DT, dst1_data, dst1_step), dst1_step )

#define TegraBinaryOp_InvokerVAArg(name, func, ...) TegraGenOp_Invoker(name, func, 2, 1, 0, \
                                                                       RANGE_DATA(ST, src1_data, src1_step), src1_step, \
                                                                       RANGE_DATA(ST, src2_data, src2_step), src2_step, \
                                                                       RANGE_DATA(DT, dst1_data, dst1_step), dst1_step, __VA_ARGS__)

#define TEGRA_BINARYOP(type, op, src1, sz1, src2, sz2, dst, sz, w, h) \
( \
    CAROTENE_NS::isSupportedConfiguration() ? \
    parallel_for_(Range(0, h), \
    TegraGenOp_##op##_Invoker<const type, type>(src1, sz1, src2, sz2, dst, sz, w, h), \
    (w * h) / static_cast<double>(1<<16)), \
    CV_HAL_ERROR_OK \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

TegraBinaryOp_InvokerVAArg(add, add, CAROTENE_NS::CONVERT_POLICY_SATURATE) /*Original addition use saturated operator, so use the same from CAROTENE*/

TegraBinaryOp_Invoker(addf, add)

TegraBinaryOp_InvokerVAArg(sub, sub, CAROTENE_NS::CONVERT_POLICY_SATURATE) /*Original addition use saturated operator, so use the same from CAROTENE*/

TegraBinaryOp_Invoker(subf, sub)

TegraBinaryOp_Invoker(max, max)

TegraBinaryOp_Invoker(min, min)

TegraBinaryOp_Invoker(absDiff, absDiff)

TegraBinaryOp_Invoker(bitwiseAnd, bitwiseAnd)

TegraBinaryOp_Invoker(bitwiseOr, bitwiseOr)

TegraBinaryOp_Invoker(bitwiseXor, bitwiseXor)

#define TegraUnaryOp_Invoker(name, func) TegraGenOp_Invoker(name, func, 1, 1, 0, \
                                                            RANGE_DATA(ST, src1_data, src1_step), src1_step, \
                                                            RANGE_DATA(DT, dst1_data, dst1_step), dst1_step )

TegraUnaryOp_Invoker(bitwiseNot, bitwiseNot)
#define TEGRA_UNARYOP(type, op, src1, sz1, dst, sz, w, h) \
( \
    CAROTENE_NS::isSupportedConfiguration() ? \
    parallel_for_(Range(0, h), \
    TegraGenOp_##op##_Invoker<const type, type>(src1, sz1, dst, sz, w, h), \
    (w * h) / static_cast<double>(1<<16)), \
    CV_HAL_ERROR_OK \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

#undef cv_hal_add8u
#define cv_hal_add8u(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::u8, add, src1, sz1, src2, sz2, dst, sz, w, h)
#undef cv_hal_add8s
#define cv_hal_add8s(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::s8, add, src1, sz1, src2, sz2, dst, sz, w, h)
#undef cv_hal_add16u
#define cv_hal_add16u(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::u16, add, src1, sz1, src2, sz2, dst, sz, w, h)
#undef cv_hal_add16s
#define cv_hal_add16s(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::s16, add, src1, sz1, src2, sz2, dst, sz, w, h)
#undef cv_hal_add32s
#define cv_hal_add32s(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::s32, add, src1, sz1, src2, sz2, dst, sz, w, h)
#undef cv_hal_add32f
#define cv_hal_add32f(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::f32, addf, src1, sz1, src2, sz2, dst, sz, w, h)
//#undef cv_hal_add64f
//#define cv_hal_add64f(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::f64, addf, src1, sz1, src2, sz2, dst, sz, w, h)
#undef cv_hal_sub8u
#define cv_hal_sub8u(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::u8, sub, src1, sz1, src2, sz2, dst, sz, w, h)
#undef cv_hal_sub8s
#define cv_hal_sub8s(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::s8, sub, src1, sz1, src2, sz2, dst, sz, w, h)
#undef cv_hal_sub16u
#define cv_hal_sub16u(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::u16, sub, src1, sz1, src2, sz2, dst, sz, w, h)
#undef cv_hal_sub16s
#define cv_hal_sub16s(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::s16, sub, src1, sz1, src2, sz2, dst, sz, w, h)
#undef cv_hal_sub32s
#define cv_hal_sub32s(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::s32, sub, src1, sz1, src2, sz2, dst, sz, w, h)
#undef cv_hal_sub32f
#define cv_hal_sub32f(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::f32, subf, src1, sz1, src2, sz2, dst, sz, w, h)
//#undef cv_hal_sub64f
//#define cv_hal_sub64f(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::f64, subf, src1, sz1, src2, sz2, dst, sz, w, h)
#undef cv_hal_max8u
#define cv_hal_max8u(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::u8, max, src1, sz1, src2, sz2, dst, sz, w, h)
#undef cv_hal_max8s
#define cv_hal_max8s(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::s8, max, src1, sz1, src2, sz2, dst, sz, w, h)
#undef cv_hal_max16u
#define cv_hal_max16u(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::u16, max, src1, sz1, src2, sz2, dst, sz, w, h)
#undef cv_hal_max16s
#define cv_hal_max16s(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::s16, max, src1, sz1, src2, sz2, dst, sz, w, h)
#undef cv_hal_max32s
#define cv_hal_max32s(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::s32, max, src1, sz1, src2, sz2, dst, sz, w, h)
#undef cv_hal_max32f
#define cv_hal_max32f(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::f32, max, src1, sz1, src2, sz2, dst, sz, w, h)
//#undef cv_hal_max64f
//#define cv_hal_max64f(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::f64, max, src1, sz1, src2, sz2, dst, sz, w, h)
#undef cv_hal_min8u
#define cv_hal_min8u(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::u8, min, src1, sz1, src2, sz2, dst, sz, w, h)
#undef cv_hal_min8s
#define cv_hal_min8s(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::s8, min, src1, sz1, src2, sz2, dst, sz, w, h)
#undef cv_hal_min16u
#define cv_hal_min16u(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::u16, min, src1, sz1, src2, sz2, dst, sz, w, h)
#undef cv_hal_min16s
#define cv_hal_min16s(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::s16, min, src1, sz1, src2, sz2, dst, sz, w, h)
#undef cv_hal_min32s
#define cv_hal_min32s(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::s32, min, src1, sz1, src2, sz2, dst, sz, w, h)
#undef cv_hal_min32f
#define cv_hal_min32f(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::f32, min, src1, sz1, src2, sz2, dst, sz, w, h)
//#undef cv_hal_min64f
//#define cv_hal_min64f(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::f64, min, src1, sz1, src2, sz2, dst, sz, w, h)
#undef cv_hal_absdiff8u
#define cv_hal_absdiff8u(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::u8, absDiff, src1, sz1, src2, sz2, dst, sz, w, h)
#undef cv_hal_absdiff8s
#define cv_hal_absdiff8s(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::s8, absDiff, src1, sz1, src2, sz2, dst, sz, w, h)
#undef cv_hal_absdiff16u
#define cv_hal_absdiff16u(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::u16, absDiff, src1, sz1, src2, sz2, dst, sz, w, h)
#undef cv_hal_absdiff16s
#define cv_hal_absdiff16s(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::s16, absDiff, src1, sz1, src2, sz2, dst, sz, w, h)
#undef cv_hal_absdiff32s
#define cv_hal_absdiff32s(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::s32, absDiff, src1, sz1, src2, sz2, dst, sz, w, h)
#undef cv_hal_absdiff32f
#define cv_hal_absdiff32f(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::f32, absDiff, src1, sz1, src2, sz2, dst, sz, w, h)
//#undef cv_hal_absdiff64f
//#define cv_hal_absdiff64f(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::f64, absDiff, src1, sz1, src2, sz2, dst, sz, w, h)
#undef cv_hal_and8u
#define cv_hal_and8u(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::u8, bitwiseAnd, src1, sz1, src2, sz2, dst, sz, w, h)
#undef cv_hal_or8u
#define cv_hal_or8u(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::u8, bitwiseOr, src1, sz1, src2, sz2, dst, sz, w, h)
#undef cv_hal_xor8u
#define cv_hal_xor8u(src1, sz1, src2, sz2, dst, sz, w, h) TEGRA_BINARYOP(CAROTENE_NS::u8, bitwiseXor, src1, sz1, src2, sz2, dst, sz, w, h)
#undef cv_hal_not8u
#define cv_hal_not8u(src1, sz1, dst, sz, w, h) TEGRA_UNARYOP(CAROTENE_NS::u8, bitwiseNot, src1, sz1, dst, sz, w, h)

TegraBinaryOp_Invoker(cmpEQ, cmpEQ)
TegraBinaryOp_Invoker(cmpNE, cmpNE)
TegraBinaryOp_Invoker(cmpGT, cmpGT)
TegraBinaryOp_Invoker(cmpGE, cmpGE)
TegraGenOp_Invoker(cmpLT, cmpGT, 2, 1, 0, RANGE_DATA(ST, src2_data, src2_step), src2_step, \
                                          RANGE_DATA(ST, src1_data, src1_step), src1_step, \
                                          RANGE_DATA(DT, dst1_data, dst1_step), dst1_step)
TegraGenOp_Invoker(cmpLE, cmpGE, 2, 1, 0, RANGE_DATA(ST, src2_data, src2_step), src2_step, \
                                          RANGE_DATA(ST, src1_data, src1_step), src1_step, \
                                          RANGE_DATA(DT, dst1_data, dst1_step), dst1_step)
#define TEGRA_CMP(type, src1, sz1, src2, sz2, dst, sz, w, h, op) \
( \
    CAROTENE_NS::isSupportedConfiguration() ? \
        ((op) == cv::CMP_EQ) ? \
        parallel_for_(Range(0, h), \
        TegraGenOp_cmpEQ_Invoker<const type, CAROTENE_NS::u8>(src1, sz1, src2, sz2, dst, sz, w, h), \
        (w * h) / static_cast<double>(1<<16)), \
        CV_HAL_ERROR_OK : \
        ((op) == cv::CMP_NE) ? \
        parallel_for_(Range(0, h), \
        TegraGenOp_cmpNE_Invoker<const type, CAROTENE_NS::u8>(src1, sz1, src2, sz2, dst, sz, w, h), \
        (w * h) / static_cast<double>(1<<16)), \
        CV_HAL_ERROR_OK : \
        ((op) == cv::CMP_GT) ? \
        parallel_for_(Range(0, h), \
        TegraGenOp_cmpGT_Invoker<const type, CAROTENE_NS::u8>(src1, sz1, src2, sz2, dst, sz, w, h), \
        (w * h) / static_cast<double>(1<<16)), \
        CV_HAL_ERROR_OK : \
        ((op) == cv::CMP_GE) ? \
        parallel_for_(Range(0, h), \
        TegraGenOp_cmpGE_Invoker<const type, CAROTENE_NS::u8>(src1, sz1, src2, sz2, dst, sz, w, h), \
        (w * h) / static_cast<double>(1<<16)), \
        CV_HAL_ERROR_OK : \
        ((op) == cv::CMP_LT) ? \
        parallel_for_(Range(0, h), \
        TegraGenOp_cmpLT_Invoker<const type, CAROTENE_NS::u8>(src1, sz1, src2, sz2, dst, sz, w, h), \
        (w * h) / static_cast<double>(1<<16)), \
        CV_HAL_ERROR_OK : \
        ((op) == cv::CMP_LE) ? \
        parallel_for_(Range(0, h), \
        TegraGenOp_cmpLE_Invoker<const type, CAROTENE_NS::u8>(src1, sz1, src2, sz2, dst, sz, w, h), \
        (w * h) / static_cast<double>(1<<16)), \
        CV_HAL_ERROR_OK : \
        CV_HAL_ERROR_NOT_IMPLEMENTED \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

#undef cv_hal_cmp8u
#define cv_hal_cmp8u(src1, sz1, src2, sz2, dst, sz, w, h, op) TEGRA_CMP(CAROTENE_NS::u8, src1, sz1, src2, sz2, dst, sz, w, h, op)
#undef cv_hal_cmp8s
#define cv_hal_cmp8s(src1, sz1, src2, sz2, dst, sz, w, h, op) TEGRA_CMP(CAROTENE_NS::s8, src1, sz1, src2, sz2, dst, sz, w, h, op)
#undef cv_hal_cmp16u
#define cv_hal_cmp16u(src1, sz1, src2, sz2, dst, sz, w, h, op) TEGRA_CMP(CAROTENE_NS::u16, src1, sz1, src2, sz2, dst, sz, w, h, op)
#undef cv_hal_cmp16s
#define cv_hal_cmp16s(src1, sz1, src2, sz2, dst, sz, w, h, op) TEGRA_CMP(CAROTENE_NS::s16, src1, sz1, src2, sz2, dst, sz, w, h, op)
#undef cv_hal_cmp32s
#define cv_hal_cmp32s(src1, sz1, src2, sz2, dst, sz, w, h, op) TEGRA_CMP(CAROTENE_NS::s32, src1, sz1, src2, sz2, dst, sz, w, h, op)
#undef cv_hal_cmp32f
#define cv_hal_cmp32f(src1, sz1, src2, sz2, dst, sz, w, h, op) TEGRA_CMP(CAROTENE_NS::f32, src1, sz1, src2, sz2, dst, sz, w, h, op)
//#undef cv_hal_cmp64f
//#define cv_hal_cmp64f(src1, sz1, src2, sz2, dst, sz, w, h, op) TEGRA_CMP(CAROTENE_NS::f64, src1, sz1, src2, sz2, dst, sz, w, h, op)

#define TegraBinaryOpScale_Invoker(name, func, scale_cnt, ...) TegraGenOp_Invoker(name, func, 2, 1, scale_cnt, \
                                                                                  RANGE_DATA(ST, src1_data, src1_step), src1_step, \
                                                                                  RANGE_DATA(ST, src2_data, src2_step), src2_step, \
                                                                                  RANGE_DATA(DT, dst1_data, dst1_step), dst1_step, __VA_ARGS__)

#define TEGRA_BINARYOPSCALE(type, op, src1, sz1, src2, sz2, dst, sz, w, h, scales) \
( \
    CAROTENE_NS::isSupportedConfiguration() ? \
    parallel_for_(Range(0, h), \
    TegraGenOp_##op##_Invoker<const type, type>(src1, sz1, src2, sz2, dst, sz, w, h, scales), \
    (w * h) / static_cast<double>(1<<16)), \
    CV_HAL_ERROR_OK \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

TegraBinaryOpScale_Invoker(mul, mul, 1, scale, CAROTENE_NS::CONVERT_POLICY_SATURATE)

TegraBinaryOpScale_Invoker(mulf, mul, 1, scale)

TegraBinaryOpScale_Invoker(div, div, 1, scale, CAROTENE_NS::CONVERT_POLICY_SATURATE)

TegraBinaryOpScale_Invoker(divf, div, 1, scale)

#define TegraUnaryOpScale_Invoker(name, func, scale_cnt, ...) TegraGenOp_Invoker(name, func, 1, 1, scale_cnt, \
                                                                                 RANGE_DATA(ST, src1_data, src1_step), src1_step, \
                                                                                 RANGE_DATA(DT, dst1_data, dst1_step), dst1_step, __VA_ARGS__)

#define TEGRA_UNARYOPSCALE(type, op, src1, sz1, dst, sz, w, h, scales) \
( \
    CAROTENE_NS::isSupportedConfiguration() ? \
    parallel_for_(Range(0, h), \
    TegraGenOp_##op##_Invoker<const type, type>(src1, sz1, dst, sz, w, h, scales), \
    (w * h) / static_cast<double>(1<<16)), \
    CV_HAL_ERROR_OK \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

TegraUnaryOpScale_Invoker(recip, reciprocal, 1, scale, CAROTENE_NS::CONVERT_POLICY_SATURATE)

TegraUnaryOpScale_Invoker(recipf, reciprocal, 1, scale)

#undef cv_hal_mul8u
#define cv_hal_mul8u(src1, sz1, src2, sz2, dst, sz, w, h, scales) TEGRA_BINARYOPSCALE(CAROTENE_NS::u8, mul, src1, sz1, src2, sz2, dst, sz, w, h, scales)
#undef cv_hal_mul8s
#define cv_hal_mul8s(src1, sz1, src2, sz2, dst, sz, w, h, scales) TEGRA_BINARYOPSCALE(CAROTENE_NS::s8, mul, src1, sz1, src2, sz2, dst, sz, w, h, scales)
#undef cv_hal_mul16u
#define cv_hal_mul16u(src1, sz1, src2, sz2, dst, sz, w, h, scales) TEGRA_BINARYOPSCALE(CAROTENE_NS::u16, mul, src1, sz1, src2, sz2, dst, sz, w, h, scales)
#undef cv_hal_mul16s
#define cv_hal_mul16s(src1, sz1, src2, sz2, dst, sz, w, h, scales) TEGRA_BINARYOPSCALE(CAROTENE_NS::s16, mul, src1, sz1, src2, sz2, dst, sz, w, h, scales)
#undef cv_hal_mul32s
#define cv_hal_mul32s(src1, sz1, src2, sz2, dst, sz, w, h, scales) TEGRA_BINARYOPSCALE(CAROTENE_NS::s32, mul, src1, sz1, src2, sz2, dst, sz, w, h, scales)
#undef cv_hal_mul32f
#define cv_hal_mul32f(src1, sz1, src2, sz2, dst, sz, w, h, scales) TEGRA_BINARYOPSCALE(CAROTENE_NS::f32, mulf, src1, sz1, src2, sz2, dst, sz, w, h, scales)
//#undef cv_hal_mul64f
//#define cv_hal_mul64f(src1, sz1, src2, sz2, dst, sz, w, h, scales) TEGRA_BINARYOPSCALE(CAROTENE_NS::f64, mulf, src1, sz1, src2, sz2, dst, sz, w, h, scales)
#undef cv_hal_div8u
#define cv_hal_div8u(src1, sz1, src2, sz2, dst, sz, w, h, scales) TEGRA_BINARYOPSCALE(CAROTENE_NS::u8, div, src1, sz1, src2, sz2, dst, sz, w, h, scales)
#undef cv_hal_div8s
#define cv_hal_div8s(src1, sz1, src2, sz2, dst, sz, w, h, scales) TEGRA_BINARYOPSCALE(CAROTENE_NS::s8, div, src1, sz1, src2, sz2, dst, sz, w, h, scales)
#undef cv_hal_div16u
#define cv_hal_div16u(src1, sz1, src2, sz2, dst, sz, w, h, scales) TEGRA_BINARYOPSCALE(CAROTENE_NS::u16, div, src1, sz1, src2, sz2, dst, sz, w, h, scales)
#undef cv_hal_div16s
#define cv_hal_div16s(src1, sz1, src2, sz2, dst, sz, w, h, scales) TEGRA_BINARYOPSCALE(CAROTENE_NS::s16, div, src1, sz1, src2, sz2, dst, sz, w, h, scales)
#undef cv_hal_div32s
#define cv_hal_div32s(src1, sz1, src2, sz2, dst, sz, w, h, scales) TEGRA_BINARYOPSCALE(CAROTENE_NS::s32, div, src1, sz1, src2, sz2, dst, sz, w, h, scales)
#undef cv_hal_div32f
#define cv_hal_div32f(src1, sz1, src2, sz2, dst, sz, w, h, scales) TEGRA_BINARYOPSCALE(CAROTENE_NS::f32, divf, src1, sz1, src2, sz2, dst, sz, w, h, scales)
//#undef cv_hal_div64f
//#define cv_hal_div64f(src1, sz1, src2, sz2, dst, sz, w, h, scales) TEGRA_BINARYOPSCALE(CAROTENE_NS::f64, divf, src1, sz1, src2, sz2, dst, sz, w, h, scales)
#undef cv_hal_recip8u
#define cv_hal_recip8u(src1, sz1, dst, sz, w, h, scales) TEGRA_UNARYOPSCALE(CAROTENE_NS::u8, recip, src1, sz1, dst, sz, w, h, scales)
#undef cv_hal_recip8s
#define cv_hal_recip8s(src1, sz1, dst, sz, w, h, scales) TEGRA_UNARYOPSCALE(CAROTENE_NS::s8, recip, src1, sz1, dst, sz, w, h, scales)
#undef cv_hal_recip16u
#define cv_hal_recip16u(src1, sz1, dst, sz, w, h, scales) TEGRA_UNARYOPSCALE(CAROTENE_NS::u16, recip, src1, sz1, dst, sz, w, h, scales)
#undef cv_hal_recip16s
#define cv_hal_recip16s(src1, sz1, dst, sz, w, h, scales) TEGRA_UNARYOPSCALE(CAROTENE_NS::s16, recip, src1, sz1, dst, sz, w, h, scales)
#undef cv_hal_recip32s
#define cv_hal_recip32s(src1, sz1, dst, sz, w, h, scales) TEGRA_UNARYOPSCALE(CAROTENE_NS::s32, recip, src1, sz1, dst, sz, w, h, scales)
#undef cv_hal_recip32f
#define cv_hal_recip32f(src1, sz1, dst, sz, w, h, scales) TEGRA_UNARYOPSCALE(CAROTENE_NS::f32, recipf, src1, sz1, dst, sz, w, h, scales)
//#undef cv_hal_recip64f
//#define cv_hal_recip64f(src1, sz1, dst, sz, w, h, scales) TEGRA_UNARYOPSCALE(CAROTENE_NS::f64, recipf, src1, sz1, dst, sz, w, h, scales)

TegraBinaryOpScale_Invoker(addWeighted, addWeighted, 3, scales[0], scales[1], scales[2])

#undef cv_hal_addWeighted8u
#define cv_hal_addWeighted8u(src1, sz1, src2, sz2, dst, sz, w, h, scales) TEGRA_BINARYOPSCALE(CAROTENE_NS::u8, addWeighted, src1, sz1, src2, sz2, dst, sz, w, h, scales)
#undef cv_hal_addWeighted8s
#define cv_hal_addWeighted8s(src1, sz1, src2, sz2, dst, sz, w, h, scales) TEGRA_BINARYOPSCALE(CAROTENE_NS::s8, addWeighted, src1, sz1, src2, sz2, dst, sz, w, h, scales)
#undef cv_hal_addWeighted16u
#define cv_hal_addWeighted16u(src1, sz1, src2, sz2, dst, sz, w, h, scales) TEGRA_BINARYOPSCALE(CAROTENE_NS::u16, addWeighted, src1, sz1, src2, sz2, dst, sz, w, h, scales)
#undef cv_hal_addWeighted16s
#define cv_hal_addWeighted16s(src1, sz1, src2, sz2, dst, sz, w, h, scales) TEGRA_BINARYOPSCALE(CAROTENE_NS::s16, addWeighted, src1, sz1, src2, sz2, dst, sz, w, h, scales)
#undef cv_hal_addWeighted32s
#define cv_hal_addWeighted32s(src1, sz1, src2, sz2, dst, sz, w, h, scales) TEGRA_BINARYOPSCALE(CAROTENE_NS::s32, addWeighted, src1, sz1, src2, sz2, dst, sz, w, h, scales)
//#undef cv_hal_addWeighted32f
//#define cv_hal_addWeighted32f(src1, sz1, src2, sz2, dst, sz, w, h, scales) TEGRA_BINARYOPSCALE(CAROTENE_NS::f32, addWeighted, src1, sz1, src2, sz2, dst, sz, w, h, scales)
//#undef cv_hal_addWeighted64f
//#define cv_hal_addWeighted64f(src1, sz1, src2, sz2, dst, sz, w, h, scales) TEGRA_BINARYOPSCALE(CAROTENE_NS::f64, addWeighted, src1, sz1, src2, sz2, dst, sz, w, h, scales)

#else

#define TEGRA_ADD(src1, sz1, src2, sz2, dst, sz, w, h) \
( \
    CAROTENE_NS::isSupportedConfiguration() ? \
    CAROTENE_NS::add(CAROTENE_NS::Size2D(w, h), \
                     src1, sz1, \
                     src2, sz2, \
                     dst, sz, \
                     CAROTENE_NS::CONVERT_POLICY_SATURATE), /*Original addition use saturated operator*/ \
                                                            /*so use the same from CAROTENE*/ \
    CV_HAL_ERROR_OK \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

#define TEGRA_ADDF(src1, sz1, src2, sz2, dst, sz, w, h) \
( \
    CAROTENE_NS::isSupportedConfiguration() ? \
    CAROTENE_NS::add(CAROTENE_NS::Size2D(w, h), \
                     src1, sz1, \
                     src2, sz2, \
                     dst, sz), \
    CV_HAL_ERROR_OK \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

#define TEGRA_SUB(src1, sz1, src2, sz2, dst, sz, w, h) \
( \
    CAROTENE_NS::isSupportedConfiguration() ? \
    CAROTENE_NS::sub(CAROTENE_NS::Size2D(w, h), \
                     src1, sz1, \
                     src2, sz2, \
                     dst, sz, \
                     CAROTENE_NS::CONVERT_POLICY_SATURATE), /*Original addition use saturated operator*/ \
                                                            /*so use the same from CAROTENE*/ \
    CV_HAL_ERROR_OK \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

#define TEGRA_SUBF(src1, sz1, src2, sz2, dst, sz, w, h) \
( \
    CAROTENE_NS::isSupportedConfiguration() ? \
    CAROTENE_NS::sub(CAROTENE_NS::Size2D(w, h), \
                     src1, sz1, \
                     src2, sz2, \
                     dst, sz), \
    CV_HAL_ERROR_OK \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

#define TEGRA_MAX(src1, sz1, src2, sz2, dst, sz, w, h) \
( \
    CAROTENE_NS::isSupportedConfiguration() ? \
    CAROTENE_NS::max(CAROTENE_NS::Size2D(w, h), \
                     src1, sz1, \
                     src2, sz2, \
                     dst, sz), \
    CV_HAL_ERROR_OK \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

#define TEGRA_MIN(src1, sz1, src2, sz2, dst, sz, w, h) \
( \
    CAROTENE_NS::isSupportedConfiguration() ? \
    CAROTENE_NS::min(CAROTENE_NS::Size2D(w, h), \
                     src1, sz1, \
                     src2, sz2, \
                     dst, sz), \
    CV_HAL_ERROR_OK \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

#define TEGRA_ABSDIFF(src1, sz1, src2, sz2, dst, sz, w, h) \
( \
    CAROTENE_NS::isSupportedConfiguration() ? \
    CAROTENE_NS::absDiff(CAROTENE_NS::Size2D(w, h), \
                     src1, sz1, \
                     src2, sz2, \
                     dst, sz), \
    CV_HAL_ERROR_OK \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

#define TEGRA_AND(src1, sz1, src2, sz2, dst, sz, w, h) \
( \
    CAROTENE_NS::isSupportedConfiguration() ? \
    CAROTENE_NS::bitwiseAnd(CAROTENE_NS::Size2D(w, h), \
                     src1, sz1, \
                     src2, sz2, \
                     dst, sz), \
    CV_HAL_ERROR_OK \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)
#define TEGRA_OR(src1, sz1, src2, sz2, dst, sz, w, h) \
( \
    CAROTENE_NS::isSupportedConfiguration() ? \
    CAROTENE_NS::bitwiseOr(CAROTENE_NS::Size2D(w, h), \
                     src1, sz1, \
                     src2, sz2, \
                     dst, sz), \
    CV_HAL_ERROR_OK \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

#define TEGRA_XOR(src1, sz1, src2, sz2, dst, sz, w, h) \
( \
    CAROTENE_NS::isSupportedConfiguration() ? \
    CAROTENE_NS::bitwiseXor(CAROTENE_NS::Size2D(w, h), \
                     src1, sz1, \
                     src2, sz2, \
                     dst, sz), \
    CV_HAL_ERROR_OK \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

#define TEGRA_NOT(src1, sz1, dst, sz, w, h) \
( \
    CAROTENE_NS::isSupportedConfiguration() ? \
    CAROTENE_NS::bitwiseNot(CAROTENE_NS::Size2D(w, h), \
                     src1, sz1, \
                     dst, sz), \
    CV_HAL_ERROR_OK \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

#undef cv_hal_add8u
#define cv_hal_add8u TEGRA_ADD
#undef cv_hal_add8s
#define cv_hal_add8s TEGRA_ADD
#undef cv_hal_add16u
#define cv_hal_add16u TEGRA_ADD
#undef cv_hal_add16s
#define cv_hal_add16s TEGRA_ADD
#undef cv_hal_add32s
#define cv_hal_add32s TEGRA_ADD
#undef cv_hal_add32f
#define cv_hal_add32f TEGRA_ADDF
//#undef cv_hal_add64f
//#define cv_hal_add64f TEGRA_ADDF
#undef cv_hal_sub8u
#define cv_hal_sub8u TEGRA_SUB
#undef cv_hal_sub8s
#define cv_hal_sub8s TEGRA_SUB
#undef cv_hal_sub16u
#define cv_hal_sub16u TEGRA_SUB
#undef cv_hal_sub16s
#define cv_hal_sub16s TEGRA_SUB
#undef cv_hal_sub32s
#define cv_hal_sub32s TEGRA_SUB
#undef cv_hal_sub32f
#define cv_hal_sub32f TEGRA_SUBF
//#undef cv_hal_sub64f
//#define cv_hal_sub64f TEGRA_SUBF
#undef cv_hal_max8u
#define cv_hal_max8u TEGRA_MAX
#undef cv_hal_max8s
#define cv_hal_max8s TEGRA_MAX
#undef cv_hal_max16u
#define cv_hal_max16u TEGRA_MAX
#undef cv_hal_max16s
#define cv_hal_max16s TEGRA_MAX
#undef cv_hal_max32s
#define cv_hal_max32s TEGRA_MAX
#undef cv_hal_max32f
#define cv_hal_max32f TEGRA_MAX
//#undef cv_hal_max64f
//#define cv_hal_max64f TEGRA_MAX
#undef cv_hal_min8u
#define cv_hal_min8u TEGRA_MIN
#undef cv_hal_min8s
#define cv_hal_min8s TEGRA_MIN
#undef cv_hal_min16u
#define cv_hal_min16u TEGRA_MIN
#undef cv_hal_min16s
#define cv_hal_min16s TEGRA_MIN
#undef cv_hal_min32s
#define cv_hal_min32s TEGRA_MIN
#undef cv_hal_min32f
#define cv_hal_min32f TEGRA_MIN
//#undef cv_hal_min64f
//#define cv_hal_min64f TEGRA_MIN
#undef cv_hal_absdiff8u
#define cv_hal_absdiff8u TEGRA_ABSDIFF
#undef cv_hal_absdiff8s
#define cv_hal_absdiff8s TEGRA_ABSDIFF
#undef cv_hal_absdiff16u
#define cv_hal_absdiff16u TEGRA_ABSDIFF
#undef cv_hal_absdiff16s
#define cv_hal_absdiff16s TEGRA_ABSDIFF
#undef cv_hal_absdiff32s
#define cv_hal_absdiff32s TEGRA_ABSDIFF
#undef cv_hal_absdiff32f
#define cv_hal_absdiff32f TEGRA_ABSDIFF
//#undef cv_hal_absdiff64f
//#define cv_hal_absdiff64f TEGRA_ABSDIFF
#undef cv_hal_and8u
#define cv_hal_and8u TEGRA_AND
#undef cv_hal_or8u
#define cv_hal_or8u TEGRA_OR
#undef cv_hal_xor8u
#define cv_hal_xor8u TEGRA_XOR
#undef cv_hal_not8u
#define cv_hal_not8u TEGRA_NOT

#define TEGRA_CMP(src1, sz1, src2, sz2, dst, sz, w, h, op) \
( \
    CAROTENE_NS::isSupportedConfiguration() ? \
        ((op) == cv::CMP_EQ) ? \
        CAROTENE_NS::cmpEQ(CAROTENE_NS::Size2D(w, h), \
                           src1, sz1, \
                           src2, sz2, \
                           dst, sz), \
        CV_HAL_ERROR_OK : \
        ((op) == cv::CMP_NE) ? \
        CAROTENE_NS::cmpNE(CAROTENE_NS::Size2D(w, h), \
                           src1, sz1, \
                           src2, sz2, \
                           dst, sz), \
        CV_HAL_ERROR_OK : \
        ((op) == cv::CMP_GT) ? \
        CAROTENE_NS::cmpGT(CAROTENE_NS::Size2D(w, h), \
                           src1, sz1, \
                           src2, sz2, \
                           dst, sz), \
        CV_HAL_ERROR_OK : \
        ((op) == cv::CMP_GE) ? \
        CAROTENE_NS::cmpGE(CAROTENE_NS::Size2D(w, h), \
                           src1, sz1, \
                           src2, sz2, \
                           dst, sz), \
        CV_HAL_ERROR_OK : \
        ((op) == cv::CMP_LT) ? \
        CAROTENE_NS::cmpGT(CAROTENE_NS::Size2D(w, h), \
                           src2, sz2, \
                           src1, sz1, \
                           dst, sz), \
        CV_HAL_ERROR_OK : \
        ((op) == cv::CMP_LE) ? \
        CAROTENE_NS::cmpGE(CAROTENE_NS::Size2D(w, h), \
                           src2, sz2, \
                           src1, sz1, \
                           dst, sz), \
        CV_HAL_ERROR_OK : \
        CV_HAL_ERROR_NOT_IMPLEMENTED \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

#undef cv_hal_cmp8u
#define cv_hal_cmp8u TEGRA_CMP
#undef cv_hal_cmp8s
#define cv_hal_cmp8s TEGRA_CMP
#undef cv_hal_cmp16u
#define cv_hal_cmp16u TEGRA_CMP
#undef cv_hal_cmp16s
#define cv_hal_cmp16s TEGRA_CMP
#undef cv_hal_cmp32s
#define cv_hal_cmp32s TEGRA_CMP
#undef cv_hal_cmp32f
#define cv_hal_cmp32f TEGRA_CMP
//#undef cv_hal_cmp64f
//#define cv_hal_cmp64f TEGRA_CMP

#define TEGRA_MUL(src1, sz1, src2, sz2, dst, sz, w, h, scale) \
( \
    CAROTENE_NS::isSupportedConfiguration() ? \
    CAROTENE_NS::mul(CAROTENE_NS::Size2D(w, h), \
                     src1, sz1, \
                     src2, sz2, \
                     dst, sz, \
                     scale, \
                     CAROTENE_NS::CONVERT_POLICY_SATURATE), \
    CV_HAL_ERROR_OK \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

#define TEGRA_MULF(src1, sz1, src2, sz2, dst, sz, w, h, scale) \
( \
    CAROTENE_NS::isSupportedConfiguration() ? \
    CAROTENE_NS::mul(CAROTENE_NS::Size2D(w, h), \
                     src1, sz1, \
                     src2, sz2, \
                     dst, sz, \
                     (float)scale), \
    CV_HAL_ERROR_OK \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

#define TEGRA_DIV(src1, sz1, src2, sz2, dst, sz, w, h, scale) \
( \
    CAROTENE_NS::isSupportedConfiguration() ? \
    CAROTENE_NS::div(CAROTENE_NS::Size2D(w, h), \
                     src1, sz1, \
                     src2, sz2, \
                     dst, sz, \
                     scale, \
                     CAROTENE_NS::CONVERT_POLICY_SATURATE), \
    CV_HAL_ERROR_OK \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

#define TEGRA_DIVF(src1, sz1, src2, sz2, dst, sz, w, h, scale) \
( \
    CAROTENE_NS::isSupportedConfiguration() ? \
    CAROTENE_NS::div(CAROTENE_NS::Size2D(w, h), \
                     src1, sz1, \
                     src2, sz2, \
                     dst, sz, \
                     (float)scale), \
    CV_HAL_ERROR_OK \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

#define TEGRA_RECIP(src2, sz2, dst, sz, w, h, scale) \
( \
    CAROTENE_NS::isSupportedConfiguration() ? \
    CAROTENE_NS::reciprocal(CAROTENE_NS::Size2D(w, h), \
                            src2, sz2, \
                            dst, sz, \
                            scale, \
                            CAROTENE_NS::CONVERT_POLICY_SATURATE), \
    CV_HAL_ERROR_OK \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

#define TEGRA_RECIPF(src2, sz2, dst, sz, w, h, scale) \
( \
    CAROTENE_NS::isSupportedConfiguration() ? \
    CAROTENE_NS::reciprocal(CAROTENE_NS::Size2D(w, h), \
                            src2, sz2, \
                            dst, sz, \
                            (float)scale), \
    CV_HAL_ERROR_OK \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

#undef cv_hal_mul8u
#define cv_hal_mul8u TEGRA_MUL
#undef cv_hal_mul8s
#define cv_hal_mul8s TEGRA_MUL
#undef cv_hal_mul16u
#define cv_hal_mul16u TEGRA_MUL
#undef cv_hal_mul16s
#define cv_hal_mul16s TEGRA_MUL
#undef cv_hal_mul32s
#define cv_hal_mul32s TEGRA_MUL
#undef cv_hal_mul32f
#define cv_hal_mul32f TEGRA_MULF
//#undef cv_hal_mul64f
//#define cv_hal_mul64f TEGRA_MULF
#undef cv_hal_div8u
#define cv_hal_div8u TEGRA_DIV
#undef cv_hal_div8s
#define cv_hal_div8s TEGRA_DIV
#undef cv_hal_div16u
#define cv_hal_div16u TEGRA_DIV
#undef cv_hal_div16s
#define cv_hal_div16s TEGRA_DIV
#undef cv_hal_div32s
#define cv_hal_div32s TEGRA_DIV
#undef cv_hal_div32f
#define cv_hal_div32f TEGRA_DIVF
//#undef cv_hal_div64f
//#define cv_hal_div64f TEGRA_DIVF
#undef cv_hal_recip8u
#define cv_hal_recip8u TEGRA_RECIP
#undef cv_hal_recip8s
#define cv_hal_recip8s TEGRA_RECIP
#undef cv_hal_recip16u
#define cv_hal_recip16u TEGRA_RECIP
#undef cv_hal_recip16s
#define cv_hal_recip16s TEGRA_RECIP
#undef cv_hal_recip32s
#define cv_hal_recip32s TEGRA_RECIP
#undef cv_hal_recip32f
#define cv_hal_recip32f TEGRA_RECIPF
//#undef cv_hal_recip64f
//#define cv_hal_recip64f TEGRA_RECIPF

#define TEGRA_ADDWEIGHTED(src1, sz1, src2, sz2, dst, sz, w, h, scales) \
( \
    CAROTENE_NS::isSupportedConfiguration() ? \
    CAROTENE_NS::addWeighted(CAROTENE_NS::Size2D(w, h), \
                             src1, sz1, \
                             src2, sz2, \
                             dst, sz, \
                             ((double *)scales)[0], ((double *)scales)[1], ((double *)scales)[2]), \
    CV_HAL_ERROR_OK \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

#undef cv_hal_addWeighted8u
#define cv_hal_addWeighted8u TEGRA_ADDWEIGHTED
#undef cv_hal_addWeighted8s
#define cv_hal_addWeighted8s TEGRA_ADDWEIGHTED
#undef cv_hal_addWeighted16u
#define cv_hal_addWeighted16u TEGRA_ADDWEIGHTED
#undef cv_hal_addWeighted16s
#define cv_hal_addWeighted16s TEGRA_ADDWEIGHTED
#undef cv_hal_addWeighted32s
#define cv_hal_addWeighted32s TEGRA_ADDWEIGHTED
//#undef cv_hal_addWeighted32f
//#define cv_hal_addWeighted32f TEGRA_ADDWEIGHTED
//#undef cv_hal_addWeighted64f
//#define cv_hal_addWeighted64f TEGRA_ADDWEIGHTED

#endif //PARALLEL_CORE

#define ROW_SRC_ARG1 const ST * src1_data_
#define ROW_SRC_STORE1 , src1_data(src1_data_)
#define ROW_SRC_VAR1 const ST * src1_data;
#define ROW_SRC_ARG2 ROW_SRC_ARG1 \
                     , const ST * src2_data_
#define ROW_SRC_STORE2 ROW_SRC_STORE1 \
                       , src2_data(src2_data_)
#define ROW_SRC_VAR2 ROW_SRC_VAR1 \
                     const ST * src2_data;
#define ROW_SRC_ARG3 ROW_SRC_ARG2 \
                     , const ST * src3_data_
#define ROW_SRC_STORE3 ROW_SRC_STORE2 \
                       , src3_data(src3_data_)
#define ROW_SRC_VAR3 ROW_SRC_VAR2 \
                     const ST * src3_data;
#define ROW_SRC_ARG4 ROW_SRC_ARG3 \
                     , const ST * src4_data_
#define ROW_SRC_STORE4 ROW_SRC_STORE3 \
                       , src4_data(src4_data_)
#define ROW_SRC_VAR4 ROW_SRC_VAR3 \
                     const ST * src4_data;

#define ROW_DST_ARG1 , DT * dst1_data_
#define ROW_DST_STORE1 , dst1_data(dst1_data_)
#define ROW_DST_VAR1 DT * dst1_data;
#define ROW_DST_ARG2 ROW_DST_ARG1 \
                     , DT * dst2_data_
#define ROW_DST_STORE2 ROW_DST_STORE1 \
                       , dst2_data(dst2_data_)
#define ROW_DST_VAR2 ROW_DST_VAR1 \
                     DT * dst2_data;
#define ROW_DST_ARG3 ROW_DST_ARG2 \
                     , DT * dst3_data_
#define ROW_DST_STORE3 ROW_DST_STORE2 \
                       , dst3_data(dst3_data_)
#define ROW_DST_VAR3 ROW_DST_VAR2 \
                     DT * dst3_data;
#define ROW_DST_ARG4 ROW_DST_ARG3 \
                     , DT * dst4_data_
#define ROW_DST_STORE4 ROW_DST_STORE3 \
                       , dst4_data(dst4_data_)
#define ROW_DST_VAR4 ROW_DST_VAR3 \
                     DT * dst4_data;

#define ROW_VAL_ARG0
#define ROW_VAL_STORE0
#define ROW_VAL_VAR0
#define ROW_VAL_ARG1 , double val_
#define ROW_VAL_STORE1 , val(val_)
#define ROW_VAL_VAR1 double val;

#define TegraRowOp_Invoker(name, func, src_cnt, dst_cnt, val_cnt, ...) \
template <typename ST, typename DT> \
class TegraRowOp_##name##_Invoker : public cv::ParallelLoopBody \
{ \
public: \
    TegraRowOp_##name##_Invoker(ROW_SRC_ARG##src_cnt \
                                ROW_DST_ARG##dst_cnt \
                                ROW_VAL_ARG##val_cnt) : \
         cv::ParallelLoopBody() ROW_SRC_STORE##src_cnt \
                                ROW_DST_STORE##dst_cnt \
                                ROW_VAL_STORE##val_cnt {} \
    virtual void operator()(const cv::Range& range) const \
    { \
        CAROTENE_NS::func(CAROTENE_NS::Size2D(range.end-range.start, 1), __VA_ARGS__); \
    } \
private: \
    ROW_SRC_VAR##src_cnt \
    ROW_DST_VAR##dst_cnt \
    ROW_VAL_VAR##val_cnt \
    const TegraRowOp_##name##_Invoker& operator= (const TegraRowOp_##name##_Invoker&); \
};


#define TEGRA_SPLIT(src, dst, len, cn) \
( \
    CAROTENE_NS::isSupportedConfiguration() ? \
        cn == 2 ? \
        CAROTENE_NS::split2(CAROTENE_NS::Size2D(len, 1), \
                            src, len, \
                            dst[0], len, \
                            dst[1], len), \
        CV_HAL_ERROR_OK : \
        cn == 3 ? \
        CAROTENE_NS::split3(CAROTENE_NS::Size2D(len, 1), \
                            src, len, \
                            dst[0], len, \
                            dst[1], len, \
                            dst[2], len), \
        CV_HAL_ERROR_OK : \
        cn == 4 ? \
        CAROTENE_NS::split4(CAROTENE_NS::Size2D(len, 1), \
                            src, len, \
                            dst[0], len, \
                            dst[1], len, \
                            dst[2], len, \
                            dst[3], len), \
        CV_HAL_ERROR_OK : \
        CV_HAL_ERROR_NOT_IMPLEMENTED \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

TegraRowOp_Invoker(split2, split2, 1, 2, 0, RANGE_DATA(ST, src1_data, 2*sizeof(ST)), range.end-range.start,
                                            RANGE_DATA(DT, dst1_data, sizeof(DT)), range.end-range.start,
                                            RANGE_DATA(DT, dst2_data, sizeof(DT)), range.end-range.start)
TegraRowOp_Invoker(split3, split3, 1, 3, 0, RANGE_DATA(ST, src1_data, 3*sizeof(ST)), range.end-range.start,
                                            RANGE_DATA(DT, dst1_data, sizeof(DT)), range.end-range.start,
                                            RANGE_DATA(DT, dst2_data, sizeof(DT)), range.end-range.start,
                                            RANGE_DATA(DT, dst3_data, sizeof(DT)), range.end-range.start)
TegraRowOp_Invoker(split4, split4, 1, 4, 0, RANGE_DATA(ST, src1_data, 4*sizeof(ST)), range.end-range.start,
                                            RANGE_DATA(DT, dst1_data, sizeof(DT)), range.end-range.start,
                                            RANGE_DATA(DT, dst2_data, sizeof(DT)), range.end-range.start,
                                            RANGE_DATA(DT, dst3_data, sizeof(DT)), range.end-range.start,
                                            RANGE_DATA(DT, dst4_data, sizeof(DT)), range.end-range.start)
#define TEGRA_SPLIT64S(type, src, dst, len, cn) \
( \
    CAROTENE_NS::isSupportedConfiguration() ? \
        cn == 2 ? \
        parallel_for_(Range(0, len), \
        TegraRowOp_split2_Invoker<const type, type>(src, dst[0], dst[1]), \
        (len) / static_cast<double>(1<<16)), \
        CV_HAL_ERROR_OK : \
        cn == 3 ? \
        parallel_for_(Range(0, len), \
        TegraRowOp_split3_Invoker<const type, type>(src, dst[0], dst[1], dst[2]), \
        (len) / static_cast<double>(1<<16)), \
        CV_HAL_ERROR_OK : \
        cn == 4 ? \
        parallel_for_(Range(0, len), \
        TegraRowOp_split4_Invoker<const type, type>(src, dst[0], dst[1], dst[2], dst[3]), \
        (len) / static_cast<double>(1<<16)), \
        CV_HAL_ERROR_OK : \
        CV_HAL_ERROR_NOT_IMPLEMENTED \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

#define TEGRA_MERGE(src, dst, len, cn) \
( \
    CAROTENE_NS::isSupportedConfiguration() ? \
        cn == 2 ? \
        CAROTENE_NS::combine2(CAROTENE_NS::Size2D(len, 1), \
                              src[0], len, \
                              src[1], len, \
                              dst, len), \
        CV_HAL_ERROR_OK : \
        cn == 3 ? \
        CAROTENE_NS::combine3(CAROTENE_NS::Size2D(len, 1), \
                              src[0], len, \
                              src[1], len, \
                              src[2], len, \
                              dst, len), \
        CV_HAL_ERROR_OK : \
        cn == 4 ? \
        CAROTENE_NS::combine4(CAROTENE_NS::Size2D(len, 1), \
                              src[0], len, \
                              src[1], len, \
                              src[2], len, \
                              src[3], len, \
                              dst, len), \
        CV_HAL_ERROR_OK : \
        CV_HAL_ERROR_NOT_IMPLEMENTED \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

TegraRowOp_Invoker(combine2, combine2, 2, 1, 0, RANGE_DATA(ST, src1_data, sizeof(ST)), range.end-range.start,
                                                RANGE_DATA(ST, src2_data, sizeof(ST)), range.end-range.start,
                                                RANGE_DATA(DT, dst1_data, 2*sizeof(DT)), range.end-range.start)
TegraRowOp_Invoker(combine3, combine3, 3, 1, 0, RANGE_DATA(ST, src1_data, sizeof(ST)), range.end-range.start,
                                                RANGE_DATA(ST, src2_data, sizeof(ST)), range.end-range.start,
                                                RANGE_DATA(ST, src3_data, sizeof(ST)), range.end-range.start,
                                                RANGE_DATA(DT, dst1_data, 3*sizeof(DT)), range.end-range.start)
TegraRowOp_Invoker(combine4, combine4, 4, 1, 0, RANGE_DATA(ST, src1_data, sizeof(ST)), range.end-range.start,
                                                RANGE_DATA(ST, src2_data, sizeof(ST)), range.end-range.start,
                                                RANGE_DATA(ST, src3_data, sizeof(ST)), range.end-range.start,
                                                RANGE_DATA(ST, src4_data, sizeof(ST)), range.end-range.start,
                                                RANGE_DATA(DT, dst1_data, 4*sizeof(DT)), range.end-range.start)
#define TEGRA_MERGE64S(type, src, dst, len, cn) \
( \
    CAROTENE_NS::isSupportedConfiguration() ? \
        cn == 2 ? \
        parallel_for_(Range(0, len), \
        TegraRowOp_combine2_Invoker<const type, type>(src[0], src[1], dst), \
        (len) / static_cast<double>(1<<16)), \
        CV_HAL_ERROR_OK : \
        cn == 3 ? \
        parallel_for_(Range(0, len), \
        TegraRowOp_combine3_Invoker<const type, type>(src[0], src[1], src[2], dst), \
        (len) / static_cast<double>(1<<16)), \
        CV_HAL_ERROR_OK : \
        cn == 4 ? \
        parallel_for_(Range(0, len), \
        TegraRowOp_combine4_Invoker<const type, type>(src[0], src[1], src[2], src[3], dst), \
        (len) / static_cast<double>(1<<16)), \
        CV_HAL_ERROR_OK : \
        CV_HAL_ERROR_NOT_IMPLEMENTED \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

#undef cv_hal_split8u
#define cv_hal_split8u TEGRA_SPLIT
#undef cv_hal_split16u
#define cv_hal_split16u TEGRA_SPLIT
#undef cv_hal_split32s
#define cv_hal_split32s TEGRA_SPLIT
#undef cv_hal_split64s
#define cv_hal_split64s(src, dst, len, cn) TEGRA_SPLIT64S(CAROTENE_NS::s64, src, dst, len, cn)

#undef cv_hal_merge8u
#define cv_hal_merge8u TEGRA_MERGE
#undef cv_hal_merge16u
#define cv_hal_merge16u TEGRA_MERGE
#undef cv_hal_merge32s
#define cv_hal_merge32s TEGRA_MERGE
#undef cv_hal_merge64s
#define cv_hal_merge64s(src, dst, len, cn) TEGRA_MERGE64S(CAROTENE_NS::s64, src, dst, len, cn)


TegraRowOp_Invoker(phase, phase, 2, 1, 1, RANGE_DATA(ST, src1_data, sizeof(CAROTENE_NS::f32)), range.end-range.start,
                                          RANGE_DATA(ST, src2_data, sizeof(CAROTENE_NS::f32)), range.end-range.start,
                                          RANGE_DATA(DT, dst1_data, sizeof(CAROTENE_NS::f32)), range.end-range.start, val)
#define TEGRA_FASTATAN(y, x, dst, len, angleInDegrees) \
( \
    CAROTENE_NS::isSupportedConfiguration() ? \
    parallel_for_(Range(0, len), \
    TegraRowOp_phase_Invoker<const CAROTENE_NS::f32, CAROTENE_NS::f32>(x, y, dst, angleInDegrees ? 1.0f : M_PI/180), \
    (len) / static_cast<double>(1<<16)), \
    CV_HAL_ERROR_OK \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

#undef cv_hal_fastAtan32f
#define cv_hal_fastAtan32f TEGRA_FASTATAN

TegraRowOp_Invoker(magnitude, magnitude, 2, 1, 0, RANGE_DATA(ST, src1_data, sizeof(CAROTENE_NS::f32)), range.end-range.start,
                                                  RANGE_DATA(ST, src2_data, sizeof(CAROTENE_NS::f32)), range.end-range.start,
                                                  RANGE_DATA(DT, dst1_data, sizeof(CAROTENE_NS::f32)), range.end-range.start)
#define TEGRA_MAGNITUDE(x, y, dst, len) \
( \
    CAROTENE_NS::isSupportedConfiguration() ? \
    parallel_for_(Range(0, len), \
    TegraRowOp_magnitude_Invoker<const CAROTENE_NS::f32, CAROTENE_NS::f32>(x, y, dst), \
    (len) / static_cast<double>(1<<16)), \
    CV_HAL_ERROR_OK \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

#undef cv_hal_magnitude32f
#define cv_hal_magnitude32f TEGRA_MAGNITUDE


#if defined OPENCV_IMGPROC_HAL_INTERFACE_H

struct cvhalFilter2D;

struct FilterCtx
{
    CAROTENE_NS::Size2D ksize;
    CAROTENE_NS::s16* kernel_data;
    CAROTENE_NS::BORDER_MODE border;
};
inline int TEGRA_FILTERINIT(cvhalFilter2D **context, uchar *kernel_data, size_t kernel_step, int kernel_type, int kernel_width, int kernel_height,
                            int max_width, int max_height, int src_type, int dst_type, int borderType, double delta, int anchor_x, int anchor_y, bool allowSubmatrix, bool allowInplace)
{
    if(!context || !kernel_data || allowSubmatrix || allowInplace ||
       src_type != CV_8UC1 || dst_type != CV_8UC1 ||
       delta != 0 || anchor_x != kernel_width / 2 || anchor_y != kernel_height / 2 )
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    FilterCtx* ctx = new FilterCtx;
    if(!ctx)
        return CV_HAL_ERROR_UNKNOWN;
    ctx->ksize.width = kernel_width;
    ctx->ksize.height = kernel_height;
    switch(borderType)
    {
    case CV_HAL_BORDER_CONSTANT:
        ctx->border = CAROTENE_NS::BORDER_MODE_CONSTANT;
        break;
    case CV_HAL_BORDER_REPLICATE:
        ctx->border = CAROTENE_NS::BORDER_MODE_REPLICATE;
        break;
    case CV_HAL_BORDER_REFLECT:
        ctx->border = CAROTENE_NS::BORDER_MODE_REFLECT;
        break;
    case CV_HAL_BORDER_WRAP:
        ctx->border = CAROTENE_NS::BORDER_MODE_WRAP;
        break;
    case CV_HAL_BORDER_REFLECT_101:
        ctx->border = CAROTENE_NS::BORDER_MODE_REFLECT101;
        break;
    default:
        delete ctx;
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    if(!CAROTENE_NS::isConvolutionSupported(CAROTENE_NS::Size2D(max_width, max_height), ctx->ksize, ctx->border))
    {
        delete ctx;
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    ctx->kernel_data = new CAROTENE_NS::s16[kernel_width*kernel_height];
    if(!ctx->kernel_data)
        return CV_HAL_ERROR_UNKNOWN;
    switch(kernel_type)
    {
    case CV_8UC1:
        convert(ctx->ksize, (CAROTENE_NS::u8*)kernel_data, kernel_step, ctx->kernel_data, kernel_width);
        break;
    case CV_8SC1:
        convert(ctx->ksize, (CAROTENE_NS::s8*)kernel_data, kernel_step, ctx->kernel_data, kernel_width);
        break;
    case CV_16UC1:
        for(int j = 0; j < kernel_height; ++j)
        {
            std::memcpy(ctx->kernel_data + kernel_width * j, kernel_data + kernel_step * j, kernel_width * sizeof(int16_t));
        }
    default:
        delete[] ctx->kernel_data;
        delete ctx;
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    *context = (cvhalFilter2D*)(ctx);
    return CV_HAL_ERROR_OK;
}
inline int TEGRA_FILTERFREE(cvhalFilter2D *context)
{
    if(context)
    {
        if(((FilterCtx*)context)->kernel_data)
            delete[] ((FilterCtx*)context)->kernel_data;
        delete (FilterCtx*)context;
        return CV_HAL_ERROR_OK;
    }
    else
    {
        return CV_HAL_ERROR_UNKNOWN;
    }
}
#define TEGRA_FILTERIMPL(context, src_data, src_step, dst_data, dst_step, width, height, full_width, full_height, offset_x, offset_y) \
( \
    (void)full_width, (void)full_height, (void)offset_x, (void)offset_y, \
    context && CAROTENE_NS::isConvolutionSupported(CAROTENE_NS::Size2D(width, height), ((FilterCtx*)context)->ksize, ((FilterCtx*)context)->border) ? \
    CAROTENE_NS::convolution(CAROTENE_NS::Size2D(width, height), \
                             src_data, src_step, \
                             dst_data, dst_step, \
                             ((FilterCtx*)context)->border, 0, \
                             ((FilterCtx*)context)->ksize, ((FilterCtx*)context)->kernel_data, 1), \
    CV_HAL_ERROR_OK \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

#undef cv_hal_filterInit
#define cv_hal_filterInit TEGRA_FILTERINIT
#undef cv_hal_filter
#define cv_hal_filter TEGRA_FILTERIMPL
#undef cv_hal_filterFree
#define cv_hal_filterFree TEGRA_FILTERFREE


struct SepFilterCtx
{
    int16_t kernelx_data[3];
    int16_t kernely_data[3];
    CAROTENE_NS::BORDER_MODE border;
};
inline int TEGRA_SEPFILTERINIT(cvhalFilter2D **context, int src_type, int dst_type, int kernel_type,
                               uchar *kernelx_data, int kernelx_length,
                               uchar *kernely_data, int kernely_length,
                               int anchor_x, int anchor_y, double delta, int borderType)
{
    if(!context || !kernelx_data || !kernely_data || src_type != CV_8UC1 || dst_type != CV_16SC1 ||
       kernelx_length != 3 || kernely_length != 3 ||
       delta != 0 || anchor_x != 1 || anchor_y != 1)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    SepFilterCtx* ctx = new SepFilterCtx;
    if(!ctx)
        return CV_HAL_ERROR_UNKNOWN;
    switch(borderType)
    {
    case CV_HAL_BORDER_CONSTANT:
        ctx->border = CAROTENE_NS::BORDER_MODE_CONSTANT;
        break;
    case CV_HAL_BORDER_REPLICATE:
        ctx->border = CAROTENE_NS::BORDER_MODE_REPLICATE;
        break;
    case CV_HAL_BORDER_REFLECT:
        ctx->border = CAROTENE_NS::BORDER_MODE_REFLECT;
        break;
    case CV_HAL_BORDER_WRAP:
        ctx->border = CAROTENE_NS::BORDER_MODE_WRAP;
        break;
    case CV_HAL_BORDER_REFLECT_101:
        ctx->border = CAROTENE_NS::BORDER_MODE_REFLECT101;
        break;
    default:
        delete ctx;
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    if(!CAROTENE_NS::isSeparableFilter3x3Supported(CAROTENE_NS::Size2D(16, 16), ctx->border, 3, 3))
    {
        delete ctx;
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    switch(kernel_type)
    {
    case CV_8UC1:
        ctx->kernelx_data[0]=kernelx_data[0];
        ctx->kernelx_data[1]=kernelx_data[1];
        ctx->kernelx_data[2]=kernelx_data[2];
        ctx->kernely_data[0]=kernely_data[0];
        ctx->kernely_data[1]=kernely_data[1];
        ctx->kernely_data[2]=kernely_data[2];
        break;
    case CV_8SC1:
        ctx->kernelx_data[0]=((char*)kernelx_data)[0];
        ctx->kernelx_data[1]=((char*)kernelx_data)[1];
        ctx->kernelx_data[2]=((char*)kernelx_data)[2];
        ctx->kernely_data[0]=((char*)kernely_data)[0];
        ctx->kernely_data[1]=((char*)kernely_data)[1];
        ctx->kernely_data[2]=((char*)kernely_data)[2];
        break;
    case CV_16UC1:
        ctx->kernelx_data[0]=((int16_t*)kernelx_data)[0];
        ctx->kernelx_data[1]=((int16_t*)kernelx_data)[1];
        ctx->kernelx_data[2]=((int16_t*)kernelx_data)[2];
        ctx->kernely_data[0]=((int16_t*)kernely_data)[0];
        ctx->kernely_data[1]=((int16_t*)kernely_data)[1];
        ctx->kernely_data[2]=((int16_t*)kernely_data)[2];
    default:
        delete ctx;
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    *context = (cvhalFilter2D*)(ctx);
    return CV_HAL_ERROR_OK;
}
inline int TEGRA_SEPFILTERFREE(cvhalFilter2D *context)
{
    if(context)
    {
        delete (SepFilterCtx*)context;
        return CV_HAL_ERROR_OK;
    }
    else
    {
        return CV_HAL_ERROR_UNKNOWN;
    }
}
#define TEGRA_SEPFILTERIMPL(context, src_data, src_step, dst_data, dst_step, width, height, full_width, full_height, offset_x, offset_y) \
( \
    context && CAROTENE_NS::isSeparableFilter3x3Supported(CAROTENE_NS::Size2D(width, height), ((SepFilterCtx*)context)->border, 3, 3, \
                                               CAROTENE_NS::Margin(offset_x, full_width - width - offset_x, offset_y, full_height - height - offset_y)) ? \
    CAROTENE_NS::SeparableFilter3x3(CAROTENE_NS::Size2D(width, height), \
                                    src_data, src_step, \
                                    (CAROTENE_NS::s16*)dst_data, dst_step, \
                                    3, 3, ((SepFilterCtx*)context)->kernelx_data, ((SepFilterCtx*)context)->kernely_data, \
                                    ((SepFilterCtx*)context)->border, 0, \
                                    CAROTENE_NS::Margin(offset_x, full_width - width - offset_x, offset_y, full_height - height - offset_y)), \
    CV_HAL_ERROR_OK \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

#undef cv_hal_sepFilterInit
#define cv_hal_sepFilterInit TEGRA_SEPFILTERINIT
#undef cv_hal_sepFilter
#define cv_hal_sepFilter TEGRA_SEPFILTERIMPL
#undef cv_hal_sepFilterFree
#define cv_hal_sepFilterFree TEGRA_SEPFILTERFREE


struct MorphCtx
{
    int operation;
    int channels;
    CAROTENE_NS::Size2D ksize;
    int anchor_x, anchor_y;
    CAROTENE_NS::BORDER_MODE border;
    uchar borderValues[4];
};
inline int TEGRA_MORPHINIT(cvhalFilter2D **context, int operation, int src_type, int dst_type, int, int,
                           int kernel_type, uchar *kernel_data, size_t kernel_step, int kernel_width, int kernel_height, int anchor_x, int anchor_y,
                           int borderType, const double borderValue[4], int iterations, bool allowSubmatrix, bool allowInplace)
{
    if(!context || !kernel_data || src_type != dst_type ||
       CV_MAT_DEPTH(src_type) != CV_8U || src_type < 0 || (src_type >> CV_CN_SHIFT) > 3 ||

       allowSubmatrix || allowInplace || iterations != 1 ||
       !CAROTENE_NS::isSupportedConfiguration())
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    switch(CV_MAT_DEPTH(kernel_type))
    {
    case CV_8U:
        if(CAROTENE_NS::countNonZero(CAROTENE_NS::Size2D(kernel_width, kernel_height), kernel_data, kernel_step) != kernel_width * kernel_height)
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        break;
    case CV_16U:
        if(CAROTENE_NS::countNonZero(CAROTENE_NS::Size2D(kernel_width, kernel_height), (uint16_t*)kernel_data, kernel_step) != kernel_width * kernel_height)
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        break;
    case CV_32S:
        if(CAROTENE_NS::countNonZero(CAROTENE_NS::Size2D(kernel_width, kernel_height), (int32_t*)kernel_data, kernel_step) != kernel_width * kernel_height)
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        break;
    case CV_32F:
        if(CAROTENE_NS::countNonZero(CAROTENE_NS::Size2D(kernel_width, kernel_height), (float*)kernel_data, kernel_step) != kernel_width * kernel_height)
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        break;
    case CV_64F:
        if(CAROTENE_NS::countNonZero(CAROTENE_NS::Size2D(kernel_width, kernel_height), (double*)kernel_data, kernel_step) != kernel_width * kernel_height)
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        break;
    default:
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    MorphCtx* ctx = new MorphCtx;
    if(!ctx)
        return CV_HAL_ERROR_UNKNOWN;
    ctx->channels = (src_type >> CV_CN_SHIFT) + 1;
    ctx->ksize.width = kernel_width;
    ctx->ksize.height = kernel_height;
    ctx->anchor_x = anchor_x;
    ctx->anchor_y = anchor_y;
    switch(operation)
    {
    case CV_HAL_MORPH_ERODE:
    case CV_HAL_MORPH_DILATE:
        ctx->operation = operation;
        break;
    default:
        delete ctx;
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }
    switch(borderType)
    {
    case CV_HAL_BORDER_CONSTANT:
        ctx->border = CAROTENE_NS::BORDER_MODE_CONSTANT;
        if( borderValue[0] == DBL_MAX && borderValue[1] == DBL_MAX && borderValue[2] == DBL_MAX && borderValue[3] == DBL_MAX )
        {
            if( operation == CV_HAL_MORPH_ERODE )
                for(int i = 0; i < ctx->channels; ++i)
                    ctx->borderValues[i] = (CAROTENE_NS::u8)UCHAR_MAX;
            else
                for(int i = 0; i < ctx->channels; ++i)
                    ctx->borderValues[i] = 0;
        }
        else
        {
            for(int i = 0; i < ctx->channels; ++i)
                ctx->borderValues[i] = (CAROTENE_NS::u8)cv::saturate_cast<uchar>(borderValue[i]);
        }
        break;
    case CV_HAL_BORDER_REPLICATE:
        ctx->border = CAROTENE_NS::BORDER_MODE_REPLICATE;
        break;
    case CV_HAL_BORDER_REFLECT:
        ctx->border = CAROTENE_NS::BORDER_MODE_REFLECT;
        break;
    case CV_HAL_BORDER_WRAP:
        ctx->border = CAROTENE_NS::BORDER_MODE_WRAP;
        break;
    case CV_HAL_BORDER_REFLECT_101:
        ctx->border = CAROTENE_NS::BORDER_MODE_REFLECT101;
        break;
    default:
        delete ctx;
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    *context = (cvhalFilter2D*)(ctx);
    return CV_HAL_ERROR_OK;
}
inline int TEGRA_MORPHFREE(cvhalFilter2D *context)
{
    if(context)
    {
        delete (MorphCtx*)context;
        return CV_HAL_ERROR_OK;
    }
    else
    {
        return CV_HAL_ERROR_UNKNOWN;
    }
}
#define TEGRA_MORPHIMPL(context, src_data, src_step, dst_data, dst_step, width, height, src_full_width, src_full_height, src_roi_x, src_roi_y, dst_full_width, dst_full_height, dst_roi_x, dst_roi_y) \
( \
    (void)dst_full_width, (void)dst_full_height, (void)dst_roi_x, (void)dst_roi_y, \
    context && CAROTENE_NS::isSupportedConfiguration() ? \
        ((MorphCtx*)context)->operation == CV_HAL_MORPH_ERODE ? \
        CAROTENE_NS::erode(CAROTENE_NS::Size2D(width, height), ((MorphCtx*)context)->channels, \
                           src_data, src_step, dst_data, dst_step, \
                           ((MorphCtx*)context)->ksize, ((MorphCtx*)context)->anchor_x, ((MorphCtx*)context)->anchor_y, \
                           ((MorphCtx*)context)->border, ((MorphCtx*)context)->border, ((MorphCtx*)context)->borderValues, \
                           CAROTENE_NS::Margin(src_roi_x, src_full_width - width - src_roi_x, src_roi_y, src_full_height - height - src_roi_y)), \
        CV_HAL_ERROR_OK : \
        ((MorphCtx*)context)->operation == CV_HAL_MORPH_DILATE ? \
        CAROTENE_NS::dilate(CAROTENE_NS::Size2D(width, height), ((MorphCtx*)context)->channels, \
                            src_data, src_step, dst_data, dst_step, \
                            ((MorphCtx*)context)->ksize, ((MorphCtx*)context)->anchor_x, ((MorphCtx*)context)->anchor_y, \
                            ((MorphCtx*)context)->border, ((MorphCtx*)context)->border, ((MorphCtx*)context)->borderValues, \
                            CAROTENE_NS::Margin(src_roi_x, src_full_width - width - src_roi_x, src_roi_y, src_full_height - height - src_roi_y)), \
        CV_HAL_ERROR_OK : \
        CV_HAL_ERROR_NOT_IMPLEMENTED \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

#undef cv_hal_morphInit
#define cv_hal_morphInit TEGRA_MORPHINIT
#undef cv_hal_morph
#define cv_hal_morph TEGRA_MORPHIMPL
#undef cv_hal_morphFree
#define cv_hal_morphFree TEGRA_MORPHFREE



#define TEGRA_RESIZE(src_type, src_data, src_step, src_width, src_height, dst_data, dst_step, dst_width, dst_height, inv_scale_x, inv_scale_y, interpolation) \
( \
    interpolation == CV_HAL_INTER_LINEAR ? \
        CV_MAT_DEPTH(src_type) == CV_8U && CAROTENE_NS::isResizeLinearOpenCVSupported(CAROTENE_NS::Size2D(src_width, src_height), CAROTENE_NS::Size2D(dst_width, dst_height), ((src_type >> CV_CN_SHIFT) + 1)) && \
        inv_scale_x > 0 && inv_scale_y > 0 && \
        (dst_width - 0.5)/inv_scale_x - 0.5 < src_width && (dst_height - 0.5)/inv_scale_y - 0.5 < src_height && \
        (dst_width + 0.5)/inv_scale_x + 0.5 >= src_width && (dst_height + 0.5)/inv_scale_y + 0.5 >= src_height && \
        std::abs(dst_width / inv_scale_x - src_width) < 0.1 && std::abs(dst_height / inv_scale_y - src_height) < 0.1 ? \
            CAROTENE_NS::resizeLinearOpenCV(CAROTENE_NS::Size2D(src_width, src_height), CAROTENE_NS::Size2D(dst_width, dst_height), \
                                            src_data, src_step, dst_data, dst_step, 1.0/inv_scale_x, 1.0/inv_scale_y, ((src_type >> CV_CN_SHIFT) + 1)), \
            CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED : \
    interpolation == CV_HAL_INTER_AREA ? \
        CV_MAT_DEPTH(src_type) == CV_8U && CAROTENE_NS::isResizeAreaSupported(1.0/inv_scale_x, 1.0/inv_scale_y, ((src_type >> CV_CN_SHIFT) + 1)) && \
        std::abs(dst_width / inv_scale_x - src_width) < 0.1 && std::abs(dst_height / inv_scale_y - src_height) < 0.1 ? \
            CAROTENE_NS::resizeAreaOpenCV(CAROTENE_NS::Size2D(src_width, src_height), CAROTENE_NS::Size2D(dst_width, dst_height), \
                                          src_data, src_step, dst_data, dst_step, 1.0/inv_scale_x, 1.0/inv_scale_y, ((src_type >> CV_CN_SHIFT) + 1)), \
            CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED : \
    /*nearest neighbour interpolation disabled due to rounding accuracy issues*/ \
    /*interpolation == CV_HAL_INTER_NEAREST ? \
        (src_type == CV_8UC1 || src_type == CV_8SC1) && CAROTENE_NS::isResizeNearestNeighborSupported(CAROTENE_NS::Size2D(src_width, src_height), 1) ? \
            CAROTENE_NS::resizeNearestNeighbor(CAROTENE_NS::Size2D(src_width, src_height), CAROTENE_NS::Size2D(dst_width, dst_height), \
                                               src_data, src_step, dst_data, dst_step, 1.0/inv_scale_x, 1.0/inv_scale_y, 1), \
            CV_HAL_ERROR_OK : \
        (src_type == CV_8UC3 || src_type == CV_8SC3) && CAROTENE_NS::isResizeNearestNeighborSupported(CAROTENE_NS::Size2D(src_width, src_height), 3) ? \
            CAROTENE_NS::resizeNearestNeighbor(CAROTENE_NS::Size2D(src_width, src_height), CAROTENE_NS::Size2D(dst_width, dst_height), \
                                               src_data, src_step, dst_data, dst_step, 1.0/inv_scale_x, 1.0/inv_scale_y, 3), \
            CV_HAL_ERROR_OK : \
        (src_type == CV_8UC4 || src_type == CV_8SC4 || src_type == CV_16UC2 || src_type == CV_16SC2 || src_type == CV_32SC1) && \
        CAROTENE_NS::isResizeNearestNeighborSupported(CAROTENE_NS::Size2D(src_width, src_height), 4) ? \
            CAROTENE_NS::resizeNearestNeighbor(CAROTENE_NS::Size2D(src_width, src_height), CAROTENE_NS::Size2D(dst_width, dst_height), \
                                               src_data, src_step, dst_data, dst_step, 1.0/inv_scale_x, 1.0/inv_scale_y, 4), \
            CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED :*/ \
    CV_HAL_ERROR_NOT_IMPLEMENTED \
)

#define TEGRA_WARPAFFINE(src_type, src_data, src_step, src_width, src_height, dst_data, dst_step, dst_width, dst_height, M, interpolation, borderType, borderValue) \
( \
    interpolation == CV_HAL_INTER_NEAREST ? \
        (src_type == CV_8UC1 || src_type == CV_8SC1) && (borderType == CV_HAL_BORDER_REPLICATE || borderType == CV_HAL_BORDER_CONSTANT) && \
        CAROTENE_NS::isWarpAffineNearestNeighborSupported(CAROTENE_NS::Size2D(src_width, src_height)) ? \
            CAROTENE_NS::warpAffineNearestNeighbor(CAROTENE_NS::Size2D(src_width, src_height), CAROTENE_NS::Size2D(dst_width, dst_height), \
                                                   src_data, src_step, \
                                                   std::vector<float>(M+0,M+6).data(), \
                                                   dst_data, dst_step, \
                                                   borderType == CV_HAL_BORDER_REPLICATE ? CAROTENE_NS::BORDER_MODE_REPLICATE : CAROTENE_NS::BORDER_MODE_CONSTANT, \
                                                   (CAROTENE_NS::u8)borderValue[0]), \
        CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED : \
    interpolation == CV_HAL_INTER_LINEAR ? \
        (src_type == CV_8UC1 || src_type == CV_8SC1) && (borderType == CV_HAL_BORDER_REPLICATE || borderType == CV_HAL_BORDER_CONSTANT) && \
        CAROTENE_NS::isWarpAffineLinearSupported(CAROTENE_NS::Size2D(src_width, src_height)) ? \
            CAROTENE_NS::warpAffineLinear(CAROTENE_NS::Size2D(src_width, src_height), CAROTENE_NS::Size2D(dst_width, dst_height), \
                                          src_data, src_step, \
                                          std::vector<float>(M+0,M+6).data(), \
                                          dst_data, dst_step, \
                                          borderType == CV_HAL_BORDER_REPLICATE ? CAROTENE_NS::BORDER_MODE_REPLICATE : CAROTENE_NS::BORDER_MODE_CONSTANT, \
                                          (CAROTENE_NS::u8)borderValue[0]), \
        CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED : \
    CV_HAL_ERROR_NOT_IMPLEMENTED \
)

#define TEGRA_WARPPERSPECTIVE(src_type, src_data, src_step, src_width, src_height, dst_data, dst_step, dst_width, dst_height, M, interpolation, borderType, borderValue) \
( \
    interpolation == CV_HAL_INTER_NEAREST ? \
        (src_type == CV_8UC1 || src_type == CV_8SC1) && (borderType == CV_HAL_BORDER_REPLICATE || borderType == CV_HAL_BORDER_CONSTANT) && \
        CAROTENE_NS::isWarpPerspectiveNearestNeighborSupported(CAROTENE_NS::Size2D(src_width, src_height)) ? \
            CAROTENE_NS::warpPerspectiveNearestNeighbor(CAROTENE_NS::Size2D(src_width, src_height), CAROTENE_NS::Size2D(dst_width, dst_height), \
                                                        src_data, src_step, \
                                                        std::vector<float>(M+0,M+9).data(), \
                                                        dst_data, dst_step, \
                                                        borderType == CV_HAL_BORDER_REPLICATE ? CAROTENE_NS::BORDER_MODE_REPLICATE : CAROTENE_NS::BORDER_MODE_CONSTANT, \
                                                        (CAROTENE_NS::u8)borderValue[0]), \
        CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED : \
    interpolation == CV_HAL_INTER_LINEAR ? \
        (src_type == CV_8UC1 || src_type == CV_8SC1) && (borderType == CV_HAL_BORDER_REPLICATE || borderType == CV_HAL_BORDER_CONSTANT) && \
        CAROTENE_NS::isWarpPerspectiveLinearSupported(CAROTENE_NS::Size2D(src_width, src_height)) ? \
            CAROTENE_NS::warpPerspectiveLinear(CAROTENE_NS::Size2D(src_width, src_height), CAROTENE_NS::Size2D(dst_width, dst_height), \
                                               src_data, src_step, \
                                               std::vector<float>(M+0,M+9).data(), \
                                               dst_data, dst_step, \
                                               borderType == CV_HAL_BORDER_REPLICATE ? CAROTENE_NS::BORDER_MODE_REPLICATE : CAROTENE_NS::BORDER_MODE_CONSTANT, \
                                               (CAROTENE_NS::u8)borderValue[0]), \
        CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED : \
    CV_HAL_ERROR_NOT_IMPLEMENTED \
)

#undef cv_hal_resize
#define cv_hal_resize TEGRA_RESIZE
//warpAffine/warpPerspective disabled due to rounding accuracy issue
//#undef cv_hal_warpAffine
//#define cv_hal_warpAffine TEGRA_WARPAFFINE
//#undef cv_hal_warpPerspective
//#define cv_hal_warpPerspective TEGRA_WARPPERSPECTIVE


#define TegraCvtColor_Invoker(name, func, ...) \
class TegraCvtColor_##name##_Invoker : public cv::ParallelLoopBody \
{ \
public: \
    TegraCvtColor_##name##_Invoker(const uchar * src_data_, size_t src_step_, uchar * dst_data_, size_t dst_step_, int width_, int height_) : \
        cv::ParallelLoopBody(), src_data(src_data_), src_step(src_step_), dst_data(dst_data_), dst_step(dst_step_), width(width_), height(height_) {} \
    virtual void operator()(const cv::Range& range) const \
    { \
        CAROTENE_NS::func(CAROTENE_NS::Size2D(width, range.end-range.start), __VA_ARGS__); \
    } \
private: \
    const uchar * src_data; \
    size_t src_step; \
    uchar * dst_data; \
    size_t dst_step; \
    int width, height; \
    const TegraCvtColor_##name##_Invoker& operator= (const TegraCvtColor_##name##_Invoker&); \
};

TegraCvtColor_Invoker(rgb2bgr, rgb2bgr, src_data + static_cast<size_t>(range.start) * src_step, src_step, \
                                        dst_data + static_cast<size_t>(range.start) * dst_step, dst_step)
TegraCvtColor_Invoker(rgb2bgrx, rgb2bgrx, src_data + static_cast<size_t>(range.start) * src_step, src_step, \
                                          dst_data + static_cast<size_t>(range.start) * dst_step, dst_step)
TegraCvtColor_Invoker(rgb2rgbx, rgb2rgbx, src_data + static_cast<size_t>(range.start) * src_step, src_step, \
                                          dst_data + static_cast<size_t>(range.start) * dst_step, dst_step)
TegraCvtColor_Invoker(rgbx2bgr, rgbx2bgr, src_data + static_cast<size_t>(range.start) * src_step, src_step, \
                                          dst_data + static_cast<size_t>(range.start) * dst_step, dst_step)
TegraCvtColor_Invoker(rgbx2rgb, rgbx2rgb, src_data + static_cast<size_t>(range.start) * src_step, src_step, \
                                          dst_data + static_cast<size_t>(range.start) * dst_step, dst_step)
TegraCvtColor_Invoker(rgbx2bgrx, rgbx2bgrx, src_data + static_cast<size_t>(range.start) * src_step, src_step, \
                                            dst_data + static_cast<size_t>(range.start) * dst_step, dst_step)
#define TEGRA_CVTBGRTOBGR(src_data, src_step, dst_data, dst_step, width, height, depth, scn, dcn, swapBlue) \
( \
    depth == CV_8U && CAROTENE_NS::isSupportedConfiguration() ? \
        scn == 3 ? \
            dcn == 3 ? \
                swapBlue ? \
                    parallel_for_(Range(0, height), \
                    TegraCvtColor_rgb2bgr_Invoker(src_data, src_step, dst_data, dst_step, width, height), \
                    (width * height) / static_cast<double>(1<<16)), \
                    CV_HAL_ERROR_OK : \
                    CV_HAL_ERROR_NOT_IMPLEMENTED : \
            dcn == 4 ? \
                (swapBlue ? \
                    parallel_for_(Range(0, height), \
                    TegraCvtColor_rgb2bgrx_Invoker(src_data, src_step, dst_data, dst_step, width, height), \
                    (width * height) / static_cast<double>(1<<16)) : \
                    parallel_for_(Range(0, height), \
                    TegraCvtColor_rgb2rgbx_Invoker(src_data, src_step, dst_data, dst_step, width, height), \
                    (width * height) / static_cast<double>(1<<16)) ), \
                CV_HAL_ERROR_OK : \
            CV_HAL_ERROR_NOT_IMPLEMENTED : \
        scn == 4 ? \
            dcn == 3 ? \
                (swapBlue ? \
                    parallel_for_(Range(0, height), \
                    TegraCvtColor_rgbx2bgr_Invoker(src_data, src_step, dst_data, dst_step, width, height), \
                    (width * height) / static_cast<double>(1<<16)) : \
                    parallel_for_(Range(0, height), \
                    TegraCvtColor_rgbx2rgb_Invoker(src_data, src_step, dst_data, dst_step, width, height), \
                    (width * height) / static_cast<double>(1<<16)) ), \
                CV_HAL_ERROR_OK : \
            dcn == 4 ? \
                swapBlue ? \
                    parallel_for_(Range(0, height), \
                    TegraCvtColor_rgbx2bgrx_Invoker(src_data, src_step, dst_data, dst_step, width, height), \
                    (width * height) / static_cast<double>(1<<16)), \
                    CV_HAL_ERROR_OK : \
                    CV_HAL_ERROR_NOT_IMPLEMENTED : \
            CV_HAL_ERROR_NOT_IMPLEMENTED : \
        CV_HAL_ERROR_NOT_IMPLEMENTED \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

TegraCvtColor_Invoker(rgb2bgr565, rgb2bgr565, src_data + static_cast<size_t>(range.start) * src_step, src_step, \
                                              dst_data + static_cast<size_t>(range.start) * dst_step, dst_step)
TegraCvtColor_Invoker(rgb2rgb565, rgb2rgb565, src_data + static_cast<size_t>(range.start) * src_step, src_step, \
                                              dst_data + static_cast<size_t>(range.start) * dst_step, dst_step)
TegraCvtColor_Invoker(rgbx2bgr565, rgbx2bgr565, src_data + static_cast<size_t>(range.start) * src_step, src_step, \
                                                dst_data + static_cast<size_t>(range.start) * dst_step, dst_step)
TegraCvtColor_Invoker(rgbx2rgb565, rgbx2rgb565, src_data + static_cast<size_t>(range.start) * src_step, src_step, \
                                                dst_data + static_cast<size_t>(range.start) * dst_step, dst_step)
#define TEGRA_CVTBGRTOBGR565(src_data, src_step, dst_data, dst_step, width, height, scn, swapBlue, greenBits) \
( \
    greenBits == 6 && CAROTENE_NS::isSupportedConfiguration() ? \
        scn == 3 ? \
            (swapBlue ? \
                parallel_for_(Range(0, height), \
                TegraCvtColor_rgb2bgr565_Invoker(src_data, src_step, dst_data, dst_step, width, height), \
                (width * height) / static_cast<double>(1<<16)) : \
                parallel_for_(Range(0, height), \
                TegraCvtColor_rgb2rgb565_Invoker(src_data, src_step, dst_data, dst_step, width, height), \
                (width * height) / static_cast<double>(1<<16)) ), \
            CV_HAL_ERROR_OK : \
        scn == 4 ? \
            (swapBlue ? \
                parallel_for_(Range(0, height), \
                TegraCvtColor_rgbx2bgr565_Invoker(src_data, src_step, dst_data, dst_step, width, height), \
                (width * height) / static_cast<double>(1<<16)) : \
                parallel_for_(Range(0, height), \
                TegraCvtColor_rgbx2rgb565_Invoker(src_data, src_step, dst_data, dst_step, width, height), \
                (width * height) / static_cast<double>(1<<16)) ), \
            CV_HAL_ERROR_OK : \
        CV_HAL_ERROR_NOT_IMPLEMENTED \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

TegraCvtColor_Invoker(rgb2gray, rgb2gray, CAROTENE_NS::COLOR_SPACE_BT601, src_data + static_cast<size_t>(range.start) * src_step, src_step, \
                                                                          dst_data + static_cast<size_t>(range.start) * dst_step, dst_step)
TegraCvtColor_Invoker(bgr2gray, bgr2gray, CAROTENE_NS::COLOR_SPACE_BT601, src_data + static_cast<size_t>(range.start) * src_step, src_step, \
                                                                          dst_data + static_cast<size_t>(range.start) * dst_step, dst_step)
TegraCvtColor_Invoker(rgbx2gray, rgbx2gray, CAROTENE_NS::COLOR_SPACE_BT601, src_data + static_cast<size_t>(range.start) * src_step, src_step, \
                                                                            dst_data + static_cast<size_t>(range.start) * dst_step, dst_step)
TegraCvtColor_Invoker(bgrx2gray, bgrx2gray, CAROTENE_NS::COLOR_SPACE_BT601, src_data + static_cast<size_t>(range.start) * src_step, src_step, \
                                                                            dst_data + static_cast<size_t>(range.start) * dst_step, dst_step)
#define TEGRA_CVTBGRTOGRAY(src_data, src_step, dst_data, dst_step, width, height, depth, scn, swapBlue) \
( \
    depth == CV_8U && CAROTENE_NS::isSupportedConfiguration() ? \
        scn == 3 ? \
            (swapBlue ? \
                parallel_for_(Range(0, height), \
                TegraCvtColor_rgb2gray_Invoker(src_data, src_step, dst_data, dst_step, width, height), \
                (width * height) / static_cast<double>(1<<16)) : \
                parallel_for_(Range(0, height), \
                TegraCvtColor_bgr2gray_Invoker(src_data, src_step, dst_data, dst_step, width, height), \
                (width * height) / static_cast<double>(1<<16)) ), \
            CV_HAL_ERROR_OK : \
        scn == 4 ? \
            (swapBlue ? \
                parallel_for_(Range(0, height), \
                TegraCvtColor_rgbx2gray_Invoker(src_data, src_step, dst_data, dst_step, width, height), \
                (width * height) / static_cast<double>(1<<16)) : \
                parallel_for_(Range(0, height), \
                TegraCvtColor_bgrx2gray_Invoker(src_data, src_step, dst_data, dst_step, width, height), \
                (width * height) / static_cast<double>(1<<16)) ), \
            CV_HAL_ERROR_OK : \
        CV_HAL_ERROR_NOT_IMPLEMENTED \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

TegraCvtColor_Invoker(gray2rgb, gray2rgb, src_data + static_cast<size_t>(range.start) * src_step, src_step, \
                                          dst_data + static_cast<size_t>(range.start) * dst_step, dst_step)
TegraCvtColor_Invoker(gray2rgbx, gray2rgbx, src_data + static_cast<size_t>(range.start) * src_step, src_step, \
                                            dst_data + static_cast<size_t>(range.start) * dst_step, dst_step)
#define TEGRA_CVTGRAYTOBGR(src_data, src_step, dst_data, dst_step, width, height, depth, dcn) \
( \
    depth == CV_8U && CAROTENE_NS::isSupportedConfiguration() ? \
        dcn == 3 ? \
            parallel_for_(Range(0, height), \
            TegraCvtColor_gray2rgb_Invoker(src_data, src_step, dst_data, dst_step, width, height), \
            (width * height) / static_cast<double>(1<<16)), \
            CV_HAL_ERROR_OK : \
        dcn == 4 ? \
            parallel_for_(Range(0, height), \
            TegraCvtColor_gray2rgbx_Invoker(src_data, src_step, dst_data, dst_step, width, height), \
            (width * height) / static_cast<double>(1<<16)), \
            CV_HAL_ERROR_OK : \
        CV_HAL_ERROR_NOT_IMPLEMENTED \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

TegraCvtColor_Invoker(rgb2ycrcb, rgb2ycrcb, src_data + static_cast<size_t>(range.start) * src_step, src_step, \
                                            dst_data + static_cast<size_t>(range.start) * dst_step, dst_step)
TegraCvtColor_Invoker(bgr2ycrcb, bgr2ycrcb, src_data + static_cast<size_t>(range.start) * src_step, src_step, \
                                            dst_data + static_cast<size_t>(range.start) * dst_step, dst_step)
TegraCvtColor_Invoker(rgbx2ycrcb, rgbx2ycrcb, src_data + static_cast<size_t>(range.start) * src_step, src_step, \
                                              dst_data + static_cast<size_t>(range.start) * dst_step, dst_step)
TegraCvtColor_Invoker(bgrx2ycrcb, bgrx2ycrcb, src_data + static_cast<size_t>(range.start) * src_step, src_step, \
                                              dst_data + static_cast<size_t>(range.start) * dst_step, dst_step)
#define TEGRA_CVTBGRTOYUV(src_data, src_step, dst_data, dst_step, width, height, depth, scn, swapBlue, isCbCr) \
( \
    isCbCr && depth == CV_8U && CAROTENE_NS::isSupportedConfiguration() ? \
        scn == 3 ? \
            (swapBlue ? \
                parallel_for_(Range(0, height), \
                TegraCvtColor_rgb2ycrcb_Invoker(src_data, src_step, dst_data, dst_step, width, height), \
                (width * height) / static_cast<double>(1<<16)) : \
                parallel_for_(Range(0, height), \
                TegraCvtColor_bgr2ycrcb_Invoker(src_data, src_step, dst_data, dst_step, width, height), \
                (width * height) / static_cast<double>(1<<16)) ), \
            CV_HAL_ERROR_OK : \
        scn == 4 ? \
            (swapBlue ? \
                parallel_for_(Range(0, height), \
                TegraCvtColor_rgbx2ycrcb_Invoker(src_data, src_step, dst_data, dst_step, width, height), \
                (width * height) / static_cast<double>(1<<16)) : \
                parallel_for_(Range(0, height), \
                TegraCvtColor_bgrx2ycrcb_Invoker(src_data, src_step, dst_data, dst_step, width, height), \
                (width * height) / static_cast<double>(1<<16)) ), \
            CV_HAL_ERROR_OK : \
        CV_HAL_ERROR_NOT_IMPLEMENTED \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

TegraCvtColor_Invoker(rgb2hsv, rgb2hsv, src_data + static_cast<size_t>(range.start) * src_step, src_step, \
                                        dst_data + static_cast<size_t>(range.start) * dst_step, dst_step, 180)
TegraCvtColor_Invoker(bgr2hsv, bgr2hsv, src_data + static_cast<size_t>(range.start) * src_step, src_step, \
                                        dst_data + static_cast<size_t>(range.start) * dst_step, dst_step, 180)
TegraCvtColor_Invoker(rgbx2hsv, rgbx2hsv, src_data + static_cast<size_t>(range.start) * src_step, src_step, \
                                          dst_data + static_cast<size_t>(range.start) * dst_step, dst_step, 180)
TegraCvtColor_Invoker(bgrx2hsv, bgrx2hsv, src_data + static_cast<size_t>(range.start) * src_step, src_step, \
                                          dst_data + static_cast<size_t>(range.start) * dst_step, dst_step, 180)
TegraCvtColor_Invoker(rgb2hsvf, rgb2hsv, src_data + static_cast<size_t>(range.start) * src_step, src_step, \
                                         dst_data + static_cast<size_t>(range.start) * dst_step, dst_step, 256)
TegraCvtColor_Invoker(bgr2hsvf, bgr2hsv, src_data + static_cast<size_t>(range.start) * src_step, src_step, \
                                         dst_data + static_cast<size_t>(range.start) * dst_step, dst_step, 256)
TegraCvtColor_Invoker(rgbx2hsvf, rgbx2hsv, src_data + static_cast<size_t>(range.start) * src_step, src_step, \
                                           dst_data + static_cast<size_t>(range.start) * dst_step, dst_step, 256)
TegraCvtColor_Invoker(bgrx2hsvf, bgrx2hsv, src_data + static_cast<size_t>(range.start) * src_step, src_step, \
                                           dst_data + static_cast<size_t>(range.start) * dst_step, dst_step, 256)
#define TEGRA_CVTBGRTOHSV(src_data, src_step, dst_data, dst_step, width, height, depth, scn, swapBlue, isFullRange, isHSV) \
( \
    isHSV && depth == CV_8U && CAROTENE_NS::isSupportedConfiguration() ? \
        scn == 3 ? \
            (swapBlue ? \
                isFullRange ? \
                    parallel_for_(Range(0, height), \
                    TegraCvtColor_rgb2hsvf_Invoker(src_data, src_step, dst_data, dst_step, width, height), \
                    (width * height) / static_cast<double>(1<<16)) : \
                    parallel_for_(Range(0, height), \
                    TegraCvtColor_rgb2hsv_Invoker(src_data, src_step, dst_data, dst_step, width, height), \
                    (width * height) / static_cast<double>(1<<16)) : \
                isFullRange ? \
                    parallel_for_(Range(0, height), \
                    TegraCvtColor_bgr2hsvf_Invoker(src_data, src_step, dst_data, dst_step, width, height), \
                    (width * height) / static_cast<double>(1<<16)) : \
                    parallel_for_(Range(0, height), \
                    TegraCvtColor_bgr2hsv_Invoker(src_data, src_step, dst_data, dst_step, width, height), \
                    (width * height) / static_cast<double>(1<<16)) ), \
            CV_HAL_ERROR_OK : \
        scn == 4 ? \
            (swapBlue ? \
                isFullRange ? \
                    parallel_for_(Range(0, height), \
                    TegraCvtColor_rgbx2hsvf_Invoker(src_data, src_step, dst_data, dst_step, width, height), \
                    (width * height) / static_cast<double>(1<<16)) : \
                    parallel_for_(Range(0, height), \
                    TegraCvtColor_rgbx2hsv_Invoker(src_data, src_step, dst_data, dst_step, width, height), \
                    (width * height) / static_cast<double>(1<<16)) : \
                isFullRange ? \
                    parallel_for_(Range(0, height), \
                    TegraCvtColor_bgrx2hsvf_Invoker(src_data, src_step, dst_data, dst_step, width, height), \
                    (width * height) / static_cast<double>(1<<16)) : \
                    parallel_for_(Range(0, height), \
                    TegraCvtColor_bgrx2hsv_Invoker(src_data, src_step, dst_data, dst_step, width, height), \
                    (width * height) / static_cast<double>(1<<16)) ), \
            CV_HAL_ERROR_OK : \
        CV_HAL_ERROR_NOT_IMPLEMENTED \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

#define TEGRA_CVT2PYUVTOBGR(src_data, src_step, dst_data, dst_step, dst_width, dst_height, dcn, swapBlue, uIdx) \
( \
    CAROTENE_NS::isSupportedConfiguration() ? \
        dcn == 3 ? \
            uIdx == 0 ? \
                (swapBlue ? \
                    CAROTENE_NS::yuv420i2rgb(CAROTENE_NS::Size2D(dst_width, dst_height), \
                                             src_data, src_step, \
                                             src_data + src_step * dst_height, src_step, \
                                             dst_data, dst_step) : \
                    CAROTENE_NS::yuv420i2bgr(CAROTENE_NS::Size2D(dst_width, dst_height), \
                                             src_data, src_step, \
                                             src_data + src_step * dst_height, src_step, \
                                             dst_data, dst_step)), \
                CV_HAL_ERROR_OK : \
            uIdx == 1 ? \
                (swapBlue ? \
                    CAROTENE_NS::yuv420sp2rgb(CAROTENE_NS::Size2D(dst_width, dst_height), \
                                              src_data, src_step, \
                                              src_data + src_step * dst_height, src_step, \
                                              dst_data, dst_step) : \
                    CAROTENE_NS::yuv420sp2bgr(CAROTENE_NS::Size2D(dst_width, dst_height), \
                                              src_data, src_step, \
                                              src_data + src_step * dst_height, src_step, \
                                              dst_data, dst_step)), \
                CV_HAL_ERROR_OK : \
            CV_HAL_ERROR_NOT_IMPLEMENTED : \
        dcn == 4 ? \
            uIdx == 0 ? \
                (swapBlue ? \
                    CAROTENE_NS::yuv420i2rgbx(CAROTENE_NS::Size2D(dst_width, dst_height), \
                                              src_data, src_step, \
                                              src_data + src_step * dst_height, src_step, \
                                              dst_data, dst_step) : \
                    CAROTENE_NS::yuv420i2bgrx(CAROTENE_NS::Size2D(dst_width, dst_height), \
                                              src_data, src_step, \
                                              src_data + src_step * dst_height, src_step, \
                                              dst_data, dst_step)), \
                CV_HAL_ERROR_OK : \
            uIdx == 1 ? \
                (swapBlue ? \
                    CAROTENE_NS::yuv420sp2rgbx(CAROTENE_NS::Size2D(dst_width, dst_height), \
                                               src_data, src_step, \
                                               src_data + src_step * dst_height, src_step, \
                                               dst_data, dst_step) : \
                    CAROTENE_NS::yuv420sp2bgrx(CAROTENE_NS::Size2D(dst_width, dst_height), \
                                               src_data, src_step, \
                                               src_data + src_step * dst_height, src_step, \
                                               dst_data, dst_step)), \
                CV_HAL_ERROR_OK : \
            CV_HAL_ERROR_NOT_IMPLEMENTED : \
        CV_HAL_ERROR_NOT_IMPLEMENTED \
    : CV_HAL_ERROR_NOT_IMPLEMENTED \
)

#undef cv_hal_cvtBGRtoBGR
#define cv_hal_cvtBGRtoBGR TEGRA_CVTBGRTOBGR
#undef cv_hal_cvtBGRtoBGR5x5
#define cv_hal_cvtBGRtoBGR5x5 TEGRA_CVTBGRTOBGR565
#undef cv_hal_cvtBGRtoGray
#define cv_hal_cvtBGRtoGray TEGRA_CVTBGRTOGRAY
#undef cv_hal_cvtGraytoBGR
#define cv_hal_cvtGraytoBGR TEGRA_CVTGRAYTOBGR
#undef cv_hal_cvtBGRtoYUV
#define cv_hal_cvtBGRtoYUV TEGRA_CVTBGRTOYUV
#undef cv_hal_cvtBGRtoHSV
#define cv_hal_cvtBGRtoHSV TEGRA_CVTBGRTOHSV
#undef cv_hal_cvtTwoPlaneYUVtoBGR
#define cv_hal_cvtTwoPlaneYUVtoBGR TEGRA_CVT2PYUVTOBGR

#endif // OPENCV_IMGPROC_HAL_INTERFACE_H

#endif
