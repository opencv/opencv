// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_PYRAMIDS_HPP_INCLUDED
#define OPENCV_HAL_RVV_PYRAMIDS_HPP_INCLUDED

#include <riscv_vector.h>
#include "hal_rvv_1p0/types.hpp"

namespace cv { namespace cv_hal_rvv { namespace pyramids {

#undef cv_hal_pyrdown
#define cv_hal_pyrdown cv::cv_hal_rvv::pyramids::pyrDown
#undef cv_hal_pyrup
#define cv_hal_pyrup cv::cv_hal_rvv::pyramids::pyrUp

template<typename T> struct rvv;

template<> struct rvv<uchar>
{
    using T = RVV_U8M1;
    using WT = RVV_SameLen<int, T>;
    using MT = RVV_SameLen<uint, T>;

    static inline WT::VecType vcvt_T_WT(T::VecType a, size_t b) { return WT::reinterpret(MT::cast(a, b)); }
    static inline T::VecType vcvt_WT_T(WT::VecType a, int b, size_t c) { return T::cast(MT::reinterpret(__riscv_vsra(__riscv_vadd(a, 1 << (b - 1), c), b, c)), c); }
    static inline WT::VecType down0(WT::VecType vec_src0, WT::VecType vec_src1, WT::VecType vec_src2, WT::VecType vec_src3, WT::VecType vec_src4, size_t vl) {
        return __riscv_vadd(__riscv_vadd(__riscv_vadd(vec_src0, vec_src4, vl), __riscv_vadd(vec_src2, vec_src2, vl), vl),
                            __riscv_vsll(__riscv_vadd(__riscv_vadd(vec_src1, vec_src2, vl), vec_src3, vl), 2, vl), vl);
    }
    static inline WT::VecType down1(WT::VecType vec_src0, WT::VecType vec_src1, WT::VecType vec_src2, WT::VecType vec_src3, WT::VecType vec_src4, size_t vl) {
        return down0(vec_src0, vec_src1, vec_src2, vec_src3, vec_src4, vl);
    }
    static inline WT::VecType up00(WT::VecType vec_src0, WT::VecType vec_src1, WT::VecType vec_src2, size_t vl) {
        return __riscv_vadd(__riscv_vadd(vec_src0, vec_src2, vl), __riscv_vadd(__riscv_vsll(vec_src1, 2, vl), __riscv_vsll(vec_src1, 1, vl), vl), vl);
    }
    static inline WT::VecType up01(WT::VecType vec_src1, WT::VecType vec_src2, size_t vl) {
        return __riscv_vsll(__riscv_vadd(vec_src1, vec_src2, vl), 2, vl);
    }
    static inline WT::VecType up10(WT::VecType vec_src0, WT::VecType vec_src1, WT::VecType vec_src2, size_t vl) {
        return up00(vec_src0, vec_src1, vec_src2, vl);
    }
    static inline WT::VecType up11(WT::VecType vec_src1, WT::VecType vec_src2, size_t vl) {
        return up01(vec_src1, vec_src2, vl);
    }
};

template<> struct rvv<short>
{
    using T = RVV_I16M2;
    using WT = RVV_SameLen<int, T>;
    using MT = RVV_SameLen<uint, T>;

    static inline WT::VecType vcvt_T_WT(T::VecType a, size_t b) { return WT::cast(a, b); }
    static inline T::VecType vcvt_WT_T(WT::VecType a, int b, size_t c) { return T::cast(__riscv_vsra(__riscv_vadd(a, 1 << (b - 1), c), b, c), c); }
    static inline WT::VecType down0(WT::VecType vec_src0, WT::VecType vec_src1, WT::VecType vec_src2, WT::VecType vec_src3, WT::VecType vec_src4, size_t vl) {
        return __riscv_vadd(__riscv_vadd(__riscv_vadd(vec_src0, vec_src4, vl), __riscv_vadd(vec_src2, vec_src2, vl), vl),
                            __riscv_vsll(__riscv_vadd(__riscv_vadd(vec_src1, vec_src2, vl), vec_src3, vl), 2, vl), vl);
    }
    static inline WT::VecType down1(WT::VecType vec_src0, WT::VecType vec_src1, WT::VecType vec_src2, WT::VecType vec_src3, WT::VecType vec_src4, size_t vl) {
        return down0(vec_src0, vec_src1, vec_src2, vec_src3, vec_src4, vl);
    }
    static inline WT::VecType up00(WT::VecType vec_src0, WT::VecType vec_src1, WT::VecType vec_src2, size_t vl) {
        return __riscv_vadd(__riscv_vadd(vec_src0, vec_src2, vl), __riscv_vadd(__riscv_vsll(vec_src1, 2, vl), __riscv_vsll(vec_src1, 1, vl), vl), vl);
    }
    static inline WT::VecType up01(WT::VecType vec_src1, WT::VecType vec_src2, size_t vl) {
        return __riscv_vsll(__riscv_vadd(vec_src1, vec_src2, vl), 2, vl);
    }
    static inline WT::VecType up10(WT::VecType vec_src0, WT::VecType vec_src1, WT::VecType vec_src2, size_t vl) {
        return up00(vec_src0, vec_src1, vec_src2, vl);
    }
    static inline WT::VecType up11(WT::VecType vec_src1, WT::VecType vec_src2, size_t vl) {
        return up01(vec_src1, vec_src2, vl);
    }
};

template<> struct rvv<float>
{
    using T = RVV_F32M4;
    using WT = RVV_SameLen<float, T>;
    using MT = RVV_SameLen<uint, T>;

    static inline WT::VecType vcvt_T_WT(T::VecType a, size_t b) { return WT::cast(a, b); }
    static inline T::VecType vcvt_WT_T(WT::VecType a, [[maybe_unused]] int b, size_t c) { return T::cast(a, c); }
    static inline WT::VecType down0(WT::VecType vec_src0, WT::VecType vec_src1, WT::VecType vec_src2, WT::VecType vec_src3, WT::VecType vec_src4, size_t vl) {
        return __riscv_vfmadd(vec_src2, 6, __riscv_vfmadd(__riscv_vfadd(vec_src1, vec_src3, vl), 4, __riscv_vfadd(vec_src0, vec_src4, vl), vl), vl);
    }
    static inline WT::VecType down1(WT::VecType vec_src0, WT::VecType vec_src1, WT::VecType vec_src2, WT::VecType vec_src3, WT::VecType vec_src4, size_t vl) {
        return __riscv_vfmul(down0(vec_src0, vec_src1, vec_src2, vec_src3, vec_src4, vl), 1.f / 256.f, vl);
    }
    static inline WT::VecType up00(WT::VecType vec_src0, WT::VecType vec_src1, WT::VecType vec_src2, size_t vl) {
        return __riscv_vfadd(__riscv_vfmadd(vec_src1, 6, vec_src0, vl), vec_src2, vl);
    }
    static inline WT::VecType up01(WT::VecType vec_src1, WT::VecType vec_src2, size_t vl) {
        return __riscv_vfmul(__riscv_vfadd(vec_src1, vec_src2, vl), 4, vl);
    }
    static inline WT::VecType up10(WT::VecType vec_src0, WT::VecType vec_src1, WT::VecType vec_src2, size_t vl) {
        return __riscv_vfmul(__riscv_vfadd(__riscv_vfmadd(vec_src1, 6, vec_src0, vl), vec_src2, vl), 1.f / 64.f, vl);
    }
    static inline WT::VecType up11(WT::VecType vec_src1, WT::VecType vec_src2, size_t vl) {
        return __riscv_vfmul(__riscv_vfadd(vec_src1, vec_src2, vl), 1.f / 16.f, vl);
    }
};

template <typename RVV>
struct pyrDownVec0
{
    using T = typename RVV::T::ElemType;
    using WT = typename RVV::WT::ElemType;

    void operator()(const T* src, WT* row, const uint* tabM, int start, int end)
    {
        int vl;
        switch (start)
        {
        case 1:
            for( int x = start; x < end; x += vl )
            {
                vl = RVV::WT::setvl(end - x);
                auto vec_src0 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + x * 2 - 2, 2 * sizeof(T), vl), vl);
                auto vec_src1 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + x * 2 - 1, 2 * sizeof(T), vl), vl);
                auto vec_src2 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + x * 2, 2 * sizeof(T), vl), vl);
                auto vec_src3 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + x * 2 + 1, 2 * sizeof(T), vl), vl);
                auto vec_src4 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + x * 2 + 2, 2 * sizeof(T), vl), vl);
                __riscv_vse32(row + x, RVV::down0(vec_src0, vec_src1, vec_src2, vec_src3, vec_src4, vl), vl);
            }
            break;
        case 2:
            for( int x = start / 2; x < end / 2; x += vl )
            {
                vl = RVV::WT::setvl(end / 2 - x);
                auto vec_src0 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 - 2) * 2, 4 * sizeof(T), vl), vl);
                auto vec_src1 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 - 1) * 2, 4 * sizeof(T), vl), vl);
                auto vec_src2 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2) * 2, 4 * sizeof(T), vl), vl);
                auto vec_src3 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 + 1) * 2, 4 * sizeof(T), vl), vl);
                auto vec_src4 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 + 2) * 2, 4 * sizeof(T), vl), vl);
                __riscv_vsse32(row + x * 2, 2 * sizeof(WT), RVV::down0(vec_src0, vec_src1, vec_src2, vec_src3, vec_src4, vl), vl);
                vec_src0 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 - 2) * 2 + 1, 4 * sizeof(T), vl), vl);
                vec_src1 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 - 1) * 2 + 1, 4 * sizeof(T), vl), vl);
                vec_src2 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2) * 2 + 1, 4 * sizeof(T), vl), vl);
                vec_src3 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 + 1) * 2 + 1, 4 * sizeof(T), vl), vl);
                vec_src4 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 + 2) * 2 + 1, 4 * sizeof(T), vl), vl);
                __riscv_vsse32(row + x * 2 + 1, 2 * sizeof(WT), RVV::down0(vec_src0, vec_src1, vec_src2, vec_src3, vec_src4, vl), vl);
            }
            break;
        case 3:
            for( int x = start / 3; x < end / 3; x += vl )
            {
                vl = RVV::WT::setvl(end / 3 - x);
                auto vec_src0 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 - 2) * 3, 6 * sizeof(T), vl), vl);
                auto vec_src1 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 - 1) * 3, 6 * sizeof(T), vl), vl);
                auto vec_src2 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2) * 3, 6 * sizeof(T), vl), vl);
                auto vec_src3 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 + 1) * 3, 6 * sizeof(T), vl), vl);
                auto vec_src4 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 + 2) * 3, 6 * sizeof(T), vl), vl);
                __riscv_vsse32(row + x * 3, 3 * sizeof(WT), RVV::down0(vec_src0, vec_src1, vec_src2, vec_src3, vec_src4, vl), vl);
                vec_src0 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 - 2) * 3 + 1, 6 * sizeof(T), vl), vl);
                vec_src1 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 - 1) * 3 + 1, 6 * sizeof(T), vl), vl);
                vec_src2 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2) * 3 + 1, 6 * sizeof(T), vl), vl);
                vec_src3 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 + 1) * 3 + 1, 6 * sizeof(T), vl), vl);
                vec_src4 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 + 2) * 3 + 1, 6 * sizeof(T), vl), vl);
                __riscv_vsse32(row + x * 3 + 1, 3 * sizeof(WT), RVV::down0(vec_src0, vec_src1, vec_src2, vec_src3, vec_src4, vl), vl);
                vec_src0 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 - 2) * 3 + 2, 6 * sizeof(T), vl), vl);
                vec_src1 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 - 1) * 3 + 2, 6 * sizeof(T), vl), vl);
                vec_src2 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2) * 3 + 2, 6 * sizeof(T), vl), vl);
                vec_src3 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 + 1) * 3 + 2, 6 * sizeof(T), vl), vl);
                vec_src4 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 + 2) * 3 + 2, 6 * sizeof(T), vl), vl);
                __riscv_vsse32(row + x * 3 + 2, 3 * sizeof(WT), RVV::down0(vec_src0, vec_src1, vec_src2, vec_src3, vec_src4, vl), vl);
            }
            break;
        case 4:
            for( int x = start / 4; x < end / 4; x += vl )
            {
                vl = RVV::WT::setvl(end / 4 - x);
                auto vec_src0 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 - 2) * 4, 8 * sizeof(T), vl), vl);
                auto vec_src1 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 - 1) * 4, 8 * sizeof(T), vl), vl);
                auto vec_src2 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2) * 4, 8 * sizeof(T), vl), vl);
                auto vec_src3 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 + 1) * 4, 8 * sizeof(T), vl), vl);
                auto vec_src4 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 + 2) * 4, 8 * sizeof(T), vl), vl);
                __riscv_vsse32(row + x * 4, 4 * sizeof(WT), RVV::down0(vec_src0, vec_src1, vec_src2, vec_src3, vec_src4, vl), vl);
                vec_src0 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 - 2) * 4 + 1, 8 * sizeof(T), vl), vl);
                vec_src1 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 - 1) * 4 + 1, 8 * sizeof(T), vl), vl);
                vec_src2 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2) * 4 + 1, 8 * sizeof(T), vl), vl);
                vec_src3 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 + 1) * 4 + 1, 8 * sizeof(T), vl), vl);
                vec_src4 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 + 2) * 4 + 1, 8 * sizeof(T), vl), vl);
                __riscv_vsse32(row + x * 4 + 1, 4 * sizeof(WT), RVV::down0(vec_src0, vec_src1, vec_src2, vec_src3, vec_src4, vl), vl);
                vec_src0 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 - 2) * 4 + 2, 8 * sizeof(T), vl), vl);
                vec_src1 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 - 1) * 4 + 2, 8 * sizeof(T), vl), vl);
                vec_src2 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2) * 4 + 2, 8 * sizeof(T), vl), vl);
                vec_src3 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 + 1) * 4 + 2, 8 * sizeof(T), vl), vl);
                vec_src4 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 + 2) * 4 + 2, 8 * sizeof(T), vl), vl);
                __riscv_vsse32(row + x * 4 + 2, 4 * sizeof(WT), RVV::down0(vec_src0, vec_src1, vec_src2, vec_src3, vec_src4, vl), vl);
                vec_src0 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 - 2) * 4 + 3, 8 * sizeof(T), vl), vl);
                vec_src1 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 - 1) * 4 + 3, 8 * sizeof(T), vl), vl);
                vec_src2 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2) * 4 + 3, 8 * sizeof(T), vl), vl);
                vec_src3 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 + 1) * 4 + 3, 8 * sizeof(T), vl), vl);
                vec_src4 = RVV::vcvt_T_WT(RVV::T::vload_stride(src + (x * 2 + 2) * 4 + 3, 8 * sizeof(T), vl), vl);
                __riscv_vsse32(row + x * 4 + 3, 4 * sizeof(WT), RVV::down0(vec_src0, vec_src1, vec_src2, vec_src3, vec_src4, vl), vl);
            }
            break;
        default:
            for( int x = start; x < end; x += vl )
            {
                vl = RVV::WT::setvl(end - x);
                auto vec_tabM = RVV::MT::vload(tabM + x, vl);
                vec_tabM = __riscv_vmul(__riscv_vsub(vec_tabM, start * 2, vl), sizeof(T), vl);
                auto vec_src0 = RVV::vcvt_T_WT(__riscv_vloxei32(src, vec_tabM, vl), vl);
                vec_tabM =  __riscv_vadd(vec_tabM, start * sizeof(T), vl);
                auto vec_src1 = RVV::vcvt_T_WT(__riscv_vloxei32(src, vec_tabM, vl), vl);
                vec_tabM =  __riscv_vadd(vec_tabM, start * sizeof(T), vl);
                auto vec_src2 = RVV::vcvt_T_WT(__riscv_vloxei32(src, vec_tabM, vl), vl);
                vec_tabM =  __riscv_vadd(vec_tabM, start * sizeof(T), vl);
                auto vec_src3 = RVV::vcvt_T_WT(__riscv_vloxei32(src, vec_tabM, vl), vl);
                vec_tabM =  __riscv_vadd(vec_tabM, start * sizeof(T), vl);
                auto vec_src4 = RVV::vcvt_T_WT(__riscv_vloxei32(src, vec_tabM, vl), vl);
                __riscv_vse32(row + x, RVV::down0(vec_src0, vec_src1, vec_src2, vec_src3, vec_src4, vl), vl);
            }
        }
    }
};

template <typename RVV>
struct pyrDownVec1
{
    using T = typename RVV::T::ElemType;
    using WT = typename RVV::WT::ElemType;

    void operator()(WT* row0, WT* row1, WT* row2, WT* row3, WT* row4, T* dst, int end)
    {
        int vl;
        for( int x = 0 ; x < end; x += vl )
        {
            vl = RVV::WT::setvl(end - x);
            auto vec_src0 = RVV::WT::vload(row0 + x, vl);
            auto vec_src1 = RVV::WT::vload(row1 + x, vl);
            auto vec_src2 = RVV::WT::vload(row2 + x, vl);
            auto vec_src3 = RVV::WT::vload(row3 + x, vl);
            auto vec_src4 = RVV::WT::vload(row4 + x, vl);
            RVV::T::vstore(dst + x, RVV::vcvt_WT_T(RVV::down1(vec_src0, vec_src1, vec_src2, vec_src3, vec_src4, vl), 8, vl), vl);
        }
    }
};

template <typename RVV>
struct pyrUpVec0
{
    using T = typename RVV::T::ElemType;
    using WT = typename RVV::WT::ElemType;

    void operator()(const T* src, WT* row, const uint* dtab, int start, int end)
    {
        int vl;
        for( int x = start; x < end; x += vl )
        {
            vl = RVV::WT::setvl(end - x);
            auto vec_src0 = RVV::vcvt_T_WT(RVV::T::vload(src + x - start, vl), vl);
            auto vec_src1 = RVV::vcvt_T_WT(RVV::T::vload(src + x, vl), vl);
            auto vec_src2 = RVV::vcvt_T_WT(RVV::T::vload(src + x + start, vl), vl);

            auto vec_dtab = RVV::MT::vload(dtab + x, vl);
            vec_dtab = __riscv_vmul(vec_dtab, sizeof(WT), vl);
            __riscv_vsoxei32(row, vec_dtab, RVV::up00(vec_src0, vec_src1, vec_src2, vl), vl);
            __riscv_vsoxei32(row, __riscv_vadd(vec_dtab, start * sizeof(WT), vl), RVV::up01(vec_src1, vec_src2, vl), vl);
        }
    }
};

template <typename RVV>
struct pyrUpVec1
{
    using T = typename RVV::T::ElemType;
    using WT = typename RVV::WT::ElemType;

    void operator()(WT* row0, WT* row1, WT* row2, T* dst0, T* dst1, int end)
    {
        int vl;
        if (dst0 != dst1)
        {
            for( int x = 0 ; x < end; x += vl )
            {
                vl = RVV::WT::setvl(end - x);
                auto vec_src0 = RVV::WT::vload(row0 + x, vl);
                auto vec_src1 = RVV::WT::vload(row1 + x, vl);
                auto vec_src2 = RVV::WT::vload(row2 + x, vl);
                RVV::T::vstore(dst0 + x, RVV::vcvt_WT_T(RVV::up10(vec_src0, vec_src1, vec_src2, vl), 6, vl), vl);
                RVV::T::vstore(dst1 + x, RVV::vcvt_WT_T(RVV::up11(vec_src1, vec_src2, vl), 6, vl), vl);
            }
        }
        else
        {
            for( int x = 0 ; x < end; x += vl )
            {
                vl = RVV::WT::setvl(end - x);
                auto vec_src0 = RVV::WT::vload(row0 + x, vl);
                auto vec_src1 = RVV::WT::vload(row1 + x, vl);
                auto vec_src2 = RVV::WT::vload(row2 + x, vl);
                RVV::T::vstore(dst0 + x, RVV::vcvt_WT_T(RVV::up10(vec_src0, vec_src1, vec_src2, vl), 6, vl), vl);
            }
        }
    }
};

template<typename RVV>
struct PyrDownInvoker : ParallelLoopBody
{
    PyrDownInvoker(const uchar* _src_data, size_t _src_step, int _src_width, int _src_height, uchar* _dst_data, size_t _dst_step, int _dst_width, int _dst_height, int _cn, int _borderType, int* _tabR, int* _tabM, int* _tabL)
    {
        src_data = _src_data;
        src_step = _src_step;
        src_width = _src_width;
        src_height = _src_height;
        dst_data = _dst_data;
        dst_step = _dst_step;
        dst_width = _dst_width;
        dst_height = _dst_height;
        cn = _cn;
        borderType = _borderType;
        tabR = _tabR;
        tabM = _tabM;
        tabL = _tabL;
    }

    void operator()(const Range& range) const CV_OVERRIDE;

    const uchar* src_data;
    size_t src_step;
    int src_width;
    int src_height;
    uchar* dst_data;
    size_t dst_step;
    int dst_width;
    int dst_height;
    int cn;
    int borderType;
    int* tabR;
    int* tabM;
    int* tabL;
};

static inline int borderInterpolate( int p, int len, int borderType )
{
    if( (unsigned)p < (unsigned)len )
        ;
    else if( borderType == BORDER_REPLICATE )
        p = p < 0 ? 0 : len - 1;
    else if( borderType == BORDER_REFLECT || borderType == BORDER_REFLECT_101 )
    {
        int delta = borderType == BORDER_REFLECT_101;
        if( len == 1 )
            return 0;
        do
        {
            if( p < 0 )
                p = -p - 1 + delta;
            else
                p = len - 1 - (p - len) - delta;
        }
        while( (unsigned)p >= (unsigned)len );
    }
    else if( borderType == BORDER_WRAP )
    {
        if( p < 0 )
            p -= ((p-len+1)/len)*len;
        if( p >= len )
            p %= len;
    }
    else if( borderType == BORDER_CONSTANT )
        p = -1;
    return p;
}

// the algorithm is copied from imgproc/src/pyramids.cpp,
// in the function template void cv::pyrDown_

template <typename RVV>
inline int pyrDown(const uchar* src_data, size_t src_step, int src_width, int src_height, uchar* dst_data, size_t dst_step, int dst_width, int dst_height, int cn, int borderType)
{
    const int PD_SZ = 5;

    std::vector<int> _tabM(dst_width * cn), _tabL(cn * (PD_SZ + 2)), _tabR(cn * (PD_SZ + 2));
    int *tabM = _tabM.data(), *tabL = _tabL.data(), *tabR = _tabR.data();

    if( src_width <= 0 || src_height <= 0 ||
        std::abs(dst_width*2 - src_width) > 2 ||
        std::abs(dst_height*2 - src_height) > 2 )
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }
    int width0 = std::min((src_width-PD_SZ/2-1)/2 + 1, dst_width);

    for (int x = 0; x <= PD_SZ+1; x++)
    {
        int sx0 = borderInterpolate(x - PD_SZ/2, src_width, borderType)*cn;
        int sx1 = borderInterpolate(x + width0*2 - PD_SZ/2, src_width, borderType)*cn;
        for (int k = 0; k < cn; k++)
        {
            tabL[x*cn + k] = sx0 + k;
            tabR[x*cn + k] = sx1 + k;
        }
    }

    for (int x = 0; x < dst_width*cn; x++)
        tabM[x] = (x/cn)*2*cn + x % cn;

    cv::parallel_for_(Range(0,dst_height), PyrDownInvoker<RVV>(src_data, src_step, src_width, src_height, dst_data, dst_step, dst_width, dst_height, cn, borderType, tabR, tabM, tabL), cv::getNumThreads());
    return CV_HAL_ERROR_OK;
}

template <typename RVV>
void PyrDownInvoker<RVV>::operator()(const Range& range) const
{
    using T = typename RVV::T::ElemType;
    using WT = typename RVV::WT::ElemType;
    const int PD_SZ = 5;

    int bufstep = (dst_width*cn + 15) & -16;
    std::vector<WT> _buf(bufstep*PD_SZ + 16);
    WT* buf = (WT*)(((size_t)_buf.data() + 15) & -16);
    WT* rows[PD_SZ];

    int sy0 = -PD_SZ/2, sy = range.start * 2 + sy0, width0 = std::min((src_width-PD_SZ/2-1)/2 + 1, dst_width);

    int _dst_width = dst_width * cn;
    width0 *= cn;

    for (int y = range.start; y < range.end; y++)
    {
        T* dst = reinterpret_cast<T*>(dst_data + dst_step * y);
        WT *row0, *row1, *row2, *row3, *row4;

        // fill the ring buffer (horizontal convolution and decimation)
        int sy_limit = y*2 + 2;
        for( ; sy <= sy_limit; sy++ )
        {
            WT* row = buf + ((sy - sy0) % PD_SZ)*bufstep;
            int _sy = borderInterpolate(sy, src_height, borderType);
            const T* src = reinterpret_cast<const T*>(src_data + src_step * _sy);

            do {
                int x = 0;
                for( ; x < cn; x++ )
                {
                    row[x] = src[tabL[x+cn*2]]*6 + (src[tabL[x+cn]] + src[tabL[x+cn*3]])*4 +
                        src[tabL[x]] + src[tabL[x+cn*4]];
                }

                if( x == _dst_width )
                    break;

                pyrDownVec0<RVV>()(src, row, reinterpret_cast<const uint*>(tabM), cn, width0);
                x = width0;

                // tabR
                for (int x_ = 0; x < _dst_width; x++, x_++)
                {
                    row[x] = src[tabR[x_+cn*2]]*6 + (src[tabR[x_+cn]] + src[tabR[x_+cn*3]])*4 +
                        src[tabR[x_]] + src[tabR[x_+cn*4]];
                }
            } while (0);
        }

        // do vertical convolution and decimation and write the result to the destination image
        for (int k = 0; k < PD_SZ; k++)
            rows[k] = buf + ((y*2 - PD_SZ/2 + k - sy0) % PD_SZ)*bufstep;
        row0 = rows[0]; row1 = rows[1]; row2 = rows[2]; row3 = rows[3]; row4 = rows[4];

        pyrDownVec1<RVV>()(row0, row1, row2, row3, row4, dst, _dst_width);
    }
}

// the algorithm is copied from imgproc/src/pyramids.cpp,
// in the function template void cv::pyrUp_
template <typename RVV>
inline int pyrUp(const uchar* src_data, size_t src_step, int src_width, int src_height, uchar* dst_data, size_t dst_step, int dst_width, int dst_height, int cn)
{
    using T = typename RVV::T::ElemType;
    using WT = typename RVV::WT::ElemType;
    const int PU_SZ = 3;

    int bufstep = ((dst_width+1)*cn + 15) & -16;
    std::vector<WT> _buf(bufstep*PU_SZ + 16);
    WT* buf = (WT*)(((size_t)_buf.data() + 15) & -16);
    std::vector<int> _dtab(src_width*cn);
    int* dtab = _dtab.data();
    WT* rows[PU_SZ];

    if( std::abs(dst_width  - src_width*2) != dst_width % 2 ||
        std::abs(dst_height - src_height*2) != dst_height % 2)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }
    int k, x, sy0 = -PU_SZ/2, sy = sy0;

    src_width *= cn;
    dst_width *= cn;

    for( x = 0; x < src_width; x++ )
        dtab[x] = (x/cn)*2*cn + x % cn;

    for( int y = 0; y < src_height; y++ )
    {
        T* dst0 = reinterpret_cast<T*>(dst_data + dst_step * (y*2));
        T* dst1 = reinterpret_cast<T*>(dst_data + dst_step * (std::min(y*2+1, dst_height-1)));
        WT *row0, *row1, *row2;

        // fill the ring buffer (horizontal convolution and decimation)
        for( ; sy <= y + 1; sy++ )
        {
            WT* row = buf + ((sy - sy0) % PU_SZ)*bufstep;
            int _sy = borderInterpolate(sy*2, src_height*2, (int)BORDER_REFLECT_101)/2;
            const T* src = reinterpret_cast<const T*>(src_data + src_step * _sy);

            if( src_width == cn )
            {
                for( x = 0; x < cn; x++ )
                    row[x] = row[x + cn] = src[x]*8;
                continue;
            }

            for( x = 0; x < cn; x++ )
            {
                int dx = dtab[x];
                WT t0 = src[x]*6 + src[x + cn]*2;
                WT t1 = (src[x] + src[x + cn])*4;
                row[dx] = t0; row[dx + cn] = t1;
                dx = dtab[src_width - cn + x];
                int sx = src_width - cn + x;
                t0 = src[sx - cn] + src[sx]*7;
                t1 = src[sx]*8;
                row[dx] = t0; row[dx + cn] = t1;

                if (dst_width > src_width*2)
                {
                    row[(dst_width-1) * cn + x] = row[dx + cn];
                }
            }

            pyrUpVec0<RVV>()(src, row, reinterpret_cast<const uint*>(dtab), cn, src_width - cn);
        }

        // do vertical convolution and decimation and write the result to the destination image
        for( k = 0; k < PU_SZ; k++ )
            rows[k] = buf + ((y - PU_SZ/2 + k - sy0) % PU_SZ)*bufstep;
        row0 = rows[0]; row1 = rows[1]; row2 = rows[2];

        pyrUpVec1<RVV>()(row0, row1, row2, dst0, dst1, dst_width);
    }

    if (dst_height > src_height*2)
    {
        T* dst0 = reinterpret_cast<T*>(dst_data + dst_step * (src_height*2-2));
        T* dst2 = reinterpret_cast<T*>(dst_data + dst_step * (src_height*2));

        for(x = 0; x < dst_width ; x++ )
        {
            dst2[x] = dst0[x];
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int pyrDown(const uchar* src_data, size_t src_step, int src_width, int src_height, uchar* dst_data, size_t dst_step, int dst_width, int dst_height, int depth, int cn, int border_type)
{
    if (border_type == BORDER_CONSTANT || (depth == CV_32F && cn == 1))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    switch (depth)
    {
    case CV_8U:
        return pyrDown<rvv<uchar>>(src_data, src_step, src_width, src_height, dst_data, dst_step, dst_width, dst_height, cn, border_type);
    case CV_16S:
        return pyrDown<rvv<short>>(src_data, src_step, src_width, src_height, dst_data, dst_step, dst_width, dst_height, cn, border_type);
    case CV_32F:
        return pyrDown<rvv<float>>(src_data, src_step, src_width, src_height, dst_data, dst_step, dst_width, dst_height, cn, border_type);
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

inline int pyrUp(const uchar* src_data, size_t src_step, int src_width, int src_height, uchar* dst_data, size_t dst_step, int dst_width, int dst_height, int depth, int cn, int border_type)
{
    if (border_type != BORDER_DEFAULT)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    switch (depth)
    {
    case CV_8U:
        return pyrUp<rvv<uchar>>(src_data, src_step, src_width, src_height, dst_data, dst_step, dst_width, dst_height, cn);
    case CV_16S:
        return pyrUp<rvv<short>>(src_data, src_step, src_width, src_height, dst_data, dst_step, dst_width, dst_height, cn);
    case CV_32F:
        return pyrUp<rvv<float>>(src_data, src_step, src_width, src_height, dst_data, dst_step, dst_width, dst_height, cn);
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

}}}

#endif
