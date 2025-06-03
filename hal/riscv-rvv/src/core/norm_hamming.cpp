// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#include "rvv_hal.hpp"

namespace cv { namespace rvv_hal { namespace core {

#if CV_HAL_RVV_1P0_ENABLED

template <typename CellType>
inline void normHammingCnt_m8(vuint8m8_t v, vbool1_t mask, size_t len_bool, size_t& result)
{
    auto v_bool0 = __riscv_vreinterpret_b1(__riscv_vget_u8m1(v, 0));
    auto v_bool1 = __riscv_vreinterpret_b1(__riscv_vget_u8m1(v, 1));
    auto v_bool2 = __riscv_vreinterpret_b1(__riscv_vget_u8m1(v, 2));
    auto v_bool3 = __riscv_vreinterpret_b1(__riscv_vget_u8m1(v, 3));
    auto v_bool4 = __riscv_vreinterpret_b1(__riscv_vget_u8m1(v, 4));
    auto v_bool5 = __riscv_vreinterpret_b1(__riscv_vget_u8m1(v, 5));
    auto v_bool6 = __riscv_vreinterpret_b1(__riscv_vget_u8m1(v, 6));
    auto v_bool7 = __riscv_vreinterpret_b1(__riscv_vget_u8m1(v, 7));
    result += CellType::popcount(v_bool0, mask, len_bool);
    result += CellType::popcount(v_bool1, mask, len_bool);
    result += CellType::popcount(v_bool2, mask, len_bool);
    result += CellType::popcount(v_bool3, mask, len_bool);
    result += CellType::popcount(v_bool4, mask, len_bool);
    result += CellType::popcount(v_bool5, mask, len_bool);
    result += CellType::popcount(v_bool6, mask, len_bool);
    result += CellType::popcount(v_bool7, mask, len_bool);
}

template <typename CellType>
inline void normHammingCnt_m1(vuint8m1_t v, vbool1_t mask, size_t len_bool, size_t& result)
{
    auto v_bool = __riscv_vreinterpret_b1(v);
    result += CellType::popcount(v_bool, mask, len_bool);
}

struct NormHammingCell1
{
    static inline vbool1_t generateMask([[maybe_unused]] size_t len)
    {
        return vbool1_t();
    }

    template <typename T>
    static inline void preprocess([[maybe_unused]] T& v, [[maybe_unused]] size_t len)
    {
    }

    template <typename T>
    static inline size_t popcount(T v, [[maybe_unused]] vbool1_t mask, size_t len_bool)
    {
        return __riscv_vcpop(v, len_bool);
    }
};

struct NormHammingCell2
{
    static inline vbool1_t generateMask(size_t len)
    {
        return __riscv_vreinterpret_b1(__riscv_vmv_v_x_u8m1(0x55, len));
    }

    template <typename T>
    static inline void preprocess(T& v, size_t len)
    {
        v = __riscv_vor(v, __riscv_vsrl(v, 1, len), len);
    }

    template <typename T>
    static inline size_t popcount(T v, vbool1_t mask, size_t len_bool)
    {
        return __riscv_vcpop(mask, v, len_bool);
    }
};

struct NormHammingCell4
{
    static inline vbool1_t generateMask(size_t len)
    {
        return __riscv_vreinterpret_b1(__riscv_vmv_v_x_u8m1(0x11, len));
    }

    template <typename T>
    static inline void preprocess(T& v, size_t len)
    {
        v = __riscv_vor(v, __riscv_vsrl(v, 2, len), len);
        v = __riscv_vor(v, __riscv_vsrl(v, 1, len), len);
    }

    template <typename T>
    static inline size_t popcount(T v, vbool1_t mask, size_t len_bool)
    {
        return __riscv_vcpop(mask, v, len_bool);
    }
};

template <typename CellType>
inline void normHamming8uLoop(const uchar* a, size_t n, size_t& result)
{
    size_t len = __riscv_vsetvlmax_e8m8();
    size_t len_bool = len * 8;
    vbool1_t mask = CellType::generateMask(len);

    for (; n >= len; n -= len, a += len)
    {
        auto v = __riscv_vle8_v_u8m8(a, len);
        CellType::preprocess(v, len);
        normHammingCnt_m8<CellType>(v, mask, len_bool, result);
    }
    for (; n > 0; n -= len, a += len)
    {
        len = __riscv_vsetvl_e8m1(n);
        auto v = __riscv_vle8_v_u8m1(a, len);
        CellType::preprocess(v, len);
        normHammingCnt_m1<CellType>(v, mask, len * 8, result);
    }
}

template <typename CellType>
inline void normHammingDiff8uLoop(const uchar* a, const uchar* b, size_t n, size_t& result)
{
    size_t len = __riscv_vsetvlmax_e8m8();
    size_t len_bool = len * 8;
    vbool1_t mask = CellType::generateMask(len);

    for (; n >= len; n -= len, a += len, b += len)
    {
        auto v_a = __riscv_vle8_v_u8m8(a, len);
        auto v_b = __riscv_vle8_v_u8m8(b, len);
        auto v = __riscv_vxor(v_a, v_b, len);
        CellType::preprocess(v, len);
        normHammingCnt_m8<CellType>(v, mask, len_bool, result);
    }
    for (; n > 0; n -= len, a += len, b += len)
    {
        len = __riscv_vsetvl_e8m1(n);
        auto v_a = __riscv_vle8_v_u8m1(a, len);
        auto v_b = __riscv_vle8_v_u8m1(b, len);
        auto v = __riscv_vxor(v_a, v_b, len);
        CellType::preprocess(v, len);
        normHammingCnt_m1<CellType>(v, mask, len * 8, result);
    }
}

int normHamming8u(const uchar* a, int n, int cellSize, int* result)
{
    size_t _result = 0;

    switch (cellSize)
    {
        case 1: normHamming8uLoop<NormHammingCell1>(a, n, _result); break;
        case 2: normHamming8uLoop<NormHammingCell2>(a, n, _result); break;
        case 4: normHamming8uLoop<NormHammingCell4>(a, n, _result); break;
        default: return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }
    *result = static_cast<int>(_result);
    return CV_HAL_ERROR_OK;
}

int normHammingDiff8u(const uchar* a, const uchar* b, int n, int cellSize, int* result)
{
    size_t _result = 0;

    switch (cellSize)
    {
        case 1: normHammingDiff8uLoop<NormHammingCell1>(a, b, n, _result); break;
        case 2: normHammingDiff8uLoop<NormHammingCell2>(a, b, n, _result); break;
        case 4: normHammingDiff8uLoop<NormHammingCell4>(a, b, n, _result); break;
        default: return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }
    *result = static_cast<int>(_result);
    return CV_HAL_ERROR_OK;
}

#endif // CV_HAL_RVV_1P0_ENABLED

}}}  // cv::rvv_hal::core
