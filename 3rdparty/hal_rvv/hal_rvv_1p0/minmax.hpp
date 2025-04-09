// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#ifndef OPENCV_HAL_RVV_MINMAX_HPP_INCLUDED
#define OPENCV_HAL_RVV_MINMAX_HPP_INCLUDED

#include <riscv_vector.h>
#include <opencv2/core/base.hpp>
#include "hal_rvv_1p0/types.hpp"

namespace cv { namespace cv_hal_rvv { namespace minmax {

#undef cv_hal_minMaxIdx
#define cv_hal_minMaxIdx cv::cv_hal_rvv::minmax::minMaxIdx
#undef cv_hal_minMaxIdxMaskStep
#define cv_hal_minMaxIdxMaskStep cv::cv_hal_rvv::minmax::minMaxIdx

template<typename VEC_T, typename BOOL_T, typename T = typename VEC_T::ElemType>
inline int minMaxIdxReadTwice(const uchar* src_data, size_t src_step, int width, int height, double* minVal, double* maxVal,
                              int* minIdx, int* maxIdx, uchar* mask, size_t mask_step)
{
    int vlmax = VEC_T::setvlmax();
    auto vec_min = VEC_T::vmv(std::numeric_limits<T>::max(), vlmax);
    auto vec_max = VEC_T::vmv(std::numeric_limits<T>::lowest(), vlmax);
    T val_min, val_max;

    if (mask)
    {
        for (int i = 0; i < height; i++)
        {
            const T* src_row = reinterpret_cast<const T*>(src_data + i * src_step);
            const uchar* mask_row = mask + i * mask_step;
            int vl;
            for (int j = 0; j < width; j += vl)
            {
                vl = VEC_T::setvl(width - j);
                auto vec_src = VEC_T::vload(src_row + j, vl);
                auto vec_mask = BOOL_T::vload(mask_row + j, vl);
                auto bool_mask = __riscv_vmsne(vec_mask, 0, vl);
                vec_min = VEC_T::vmin_tumu(bool_mask, vec_min, vec_min, vec_src, vl);
                vec_max = VEC_T::vmax_tumu(bool_mask, vec_max, vec_max, vec_src, vl);
            }
        }

        auto sc_minval = VEC_T::vmv(std::numeric_limits<T>::max(), vlmax);
        auto sc_maxval = VEC_T::vmv(std::numeric_limits<T>::lowest(), vlmax);
        sc_minval = VEC_T::vredmin(vec_min, sc_minval, vlmax);
        sc_maxval = VEC_T::vredmax(vec_max, sc_maxval, vlmax);
        val_min = __riscv_vmv_x(sc_minval);
        val_max = __riscv_vmv_x(sc_maxval);

        bool found_min = !minIdx, found_max = !maxIdx;
        for (int i = 0; i < height && (!found_min || !found_max); i++)
        {
            const T* src_row = reinterpret_cast<const T*>(src_data + i * src_step);
            const uchar* mask_row = mask + i * mask_step;
            int vl;
            for (int j = 0; j < width && (!found_min || !found_max); j += vl)
            {
                vl = VEC_T::setvl(width - j);
                auto vec_src = VEC_T::vload(src_row + j, vl);
                auto vec_mask = BOOL_T::vload(mask_row + j, vl);
                auto bool_mask = __riscv_vmsne(vec_mask, 0, vl);
                auto bool_zero = __riscv_vmxor(bool_mask, bool_mask, vl);
                if (!found_min)
                {
                    auto bool_minpos = __riscv_vmseq_mu(bool_mask, bool_zero, vec_src, val_min, vl);
                    int index = __riscv_vfirst(bool_minpos, vl);
                    if (index != -1)
                    {
                        found_min = true;
                        minIdx[0] = i;
                        minIdx[1] = j + index;
                    }
                }
                if (!found_max)
                {
                    auto bool_maxpos = __riscv_vmseq_mu(bool_mask, bool_zero, vec_src, val_max, vl);
                    int index = __riscv_vfirst(bool_maxpos, vl);
                    if (index != -1)
                    {
                        found_max = true;
                        maxIdx[0] = i;
                        maxIdx[1] = j + index;
                    }
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < height; i++)
        {
            const T* src_row = reinterpret_cast<const T*>(src_data + i * src_step);
            int vl;
            for (int j = 0; j < width; j += vl)
            {
                vl = VEC_T::setvl(width - j);
                auto vec_src = VEC_T::vload(src_row + j, vl);
                vec_min = VEC_T::vmin_tu(vec_min, vec_min, vec_src, vl);
                vec_max = VEC_T::vmax_tu(vec_max, vec_max, vec_src, vl);
            }
        }

        auto sc_minval = VEC_T::vmv(std::numeric_limits<T>::max(), vlmax);
        auto sc_maxval = VEC_T::vmv(std::numeric_limits<T>::lowest(), vlmax);
        sc_minval = VEC_T::vredmin(vec_min, sc_minval, vlmax);
        sc_maxval = VEC_T::vredmax(vec_max, sc_maxval, vlmax);
        val_min = __riscv_vmv_x(sc_minval);
        val_max = __riscv_vmv_x(sc_maxval);

        bool found_min = !minIdx, found_max = !maxIdx;
        for (int i = 0; i < height && (!found_min || !found_max); i++)
        {
            const T* src_row = reinterpret_cast<const T*>(src_data + i * src_step);
            int vl;
            for (int j = 0; j < width && (!found_min || !found_max); j += vl)
            {
                vl = VEC_T::setvl(width - j);
                auto vec_src = VEC_T::vload(src_row + j, vl);
                if (!found_min)
                {
                    auto bool_minpos = __riscv_vmseq(vec_src, val_min, vl);
                    int index = __riscv_vfirst(bool_minpos, vl);
                    if (index != -1)
                    {
                        found_min = true;
                        minIdx[0] = i;
                        minIdx[1] = j + index;
                    }
                }
                if (!found_max)
                {
                    auto bool_maxpos = __riscv_vmseq(vec_src, val_max, vl);
                    int index = __riscv_vfirst(bool_maxpos, vl);
                    if (index != -1)
                    {
                        found_max = true;
                        maxIdx[0] = i;
                        maxIdx[1] = j + index;
                    }
                }
            }
        }
    }
    if (minVal)
    {
        *minVal = val_min;
    }
    if (maxVal)
    {
        *maxVal = val_max;
    }

    return CV_HAL_ERROR_OK;
}

template<typename VEC_T, typename BOOL_T, typename IDX_T, typename T = typename VEC_T::ElemType>
inline int minMaxIdxReadOnce(const uchar* src_data, size_t src_step, int width, int height, double* minVal, double* maxVal,
                             int* minIdx, int* maxIdx, uchar* mask, size_t mask_step)
{
    int vlmax = VEC_T::setvlmax();
    auto vec_min = VEC_T::vmv(std::numeric_limits<T>::max(), vlmax);
    auto vec_max = VEC_T::vmv(std::numeric_limits<T>::lowest(), vlmax);
    auto vec_pos = IDX_T::vid(vlmax);
    auto vec_minpos = IDX_T::vundefined(), vec_maxpos = IDX_T::vundefined();
    T val_min, val_max;

    if (mask)
    {
        for (int i = 0; i < height; i++)
        {
            const T* src_row = reinterpret_cast<const T*>(src_data + i * src_step);
            const uchar* mask_row = mask + i * mask_step;
            int vl;
            for (int j = 0; j < width; j += vl)
            {
                vl = VEC_T::setvl(width - j);
                auto vec_src = VEC_T::vload(src_row + j, vl);
                auto vec_mask = BOOL_T::vload(mask_row + j, vl);
                auto bool_mask = __riscv_vmsne(vec_mask, 0, vl);
                auto bool_zero = __riscv_vmxor(bool_mask, bool_mask, vl);

                auto bool_minpos = VEC_T::vmlt_mu(bool_mask, bool_zero, vec_src, vec_min, vl);
                auto bool_maxpos = VEC_T::vmgt_mu(bool_mask, bool_zero, vec_src, vec_max, vl);
                vec_minpos = __riscv_vmerge_tu(vec_minpos, vec_minpos, vec_pos, bool_minpos, vl);
                vec_maxpos = __riscv_vmerge_tu(vec_maxpos, vec_maxpos, vec_pos, bool_maxpos, vl);

                vec_min = __riscv_vmerge_tu(vec_min, vec_min, vec_src, bool_minpos, vl);
                vec_max = __riscv_vmerge_tu(vec_max, vec_max, vec_src, bool_maxpos, vl);
                vec_pos = __riscv_vadd(vec_pos, vl, vlmax);
            }
        }
    }
    else
    {
        for (int i = 0; i < height; i++)
        {
            const T* src_row = reinterpret_cast<const T*>(src_data + i * src_step);
            int vl;
            for (int j = 0; j < width; j += vl)
            {
                vl = VEC_T::setvl(width - j);
                auto vec_src = VEC_T::vload(src_row + j, vl);

                auto bool_minpos = VEC_T::vmlt(vec_src, vec_min, vl);
                auto bool_maxpos = VEC_T::vmgt(vec_src, vec_max, vl);
                vec_minpos = __riscv_vmerge_tu(vec_minpos, vec_minpos, vec_pos, bool_minpos, vl);
                vec_maxpos = __riscv_vmerge_tu(vec_maxpos, vec_maxpos, vec_pos, bool_maxpos, vl);

                vec_min = __riscv_vmerge_tu(vec_min, vec_min, vec_src, bool_minpos, vl);
                vec_max = __riscv_vmerge_tu(vec_max, vec_max, vec_src, bool_maxpos, vl);
                vec_pos = __riscv_vadd(vec_pos, vl, vlmax);
            }
        }
    }

    val_min = std::numeric_limits<T>::max();
    val_max = std::numeric_limits<T>::lowest();
    for (int i = 0; i < vlmax; i++)
    {
        if (val_min > VEC_T::vmv_x(vec_min))
        {
            val_min = VEC_T::vmv_x(vec_min);
            if (minIdx)
            {
                minIdx[0] = __riscv_vmv_x(vec_minpos) / width;
                minIdx[1] = __riscv_vmv_x(vec_minpos) % width;
            }
        }
        if (val_max < VEC_T::vmv_x(vec_max))
        {
            val_max = VEC_T::vmv_x(vec_max);
            if (maxIdx)
            {
                maxIdx[0] = __riscv_vmv_x(vec_maxpos) / width;
                maxIdx[1] = __riscv_vmv_x(vec_maxpos) % width;
            }
        }
        vec_min = __riscv_vslidedown(vec_min, 1, vlmax);
        vec_max = __riscv_vslidedown(vec_max, 1, vlmax);
        vec_minpos = __riscv_vslidedown(vec_minpos, 1, vlmax);
        vec_maxpos = __riscv_vslidedown(vec_maxpos, 1, vlmax);
    }
    if (minVal)
    {
        *minVal = val_min;
    }
    if (maxVal)
    {
        *maxVal = val_max;
    }

    return CV_HAL_ERROR_OK;
}

inline int minMaxIdx(const uchar* src_data, size_t src_step, int width, int height, int depth, double* minVal, double* maxVal,
                     int* minIdx, int* maxIdx, uchar* mask, size_t mask_step = 0)
{
    if (!mask_step)
        mask_step = src_step;

    switch (depth)
    {
    case CV_8UC1:
        return minMaxIdxReadTwice<RVV_U8M1, RVV_U8M1>(src_data, src_step, width, height, minVal, maxVal, minIdx, maxIdx, mask, mask_step);
    case CV_8SC1:
        return minMaxIdxReadTwice<RVV_I8M1, RVV_U8M1>(src_data, src_step, width, height, minVal, maxVal, minIdx, maxIdx, mask, mask_step);
    case CV_16UC1:
        return minMaxIdxReadTwice<RVV_U16M1, RVV_U8MF2>(src_data, src_step, width, height, minVal, maxVal, minIdx, maxIdx, mask, mask_step);
    case CV_16SC1:
        return minMaxIdxReadTwice<RVV_I16M1, RVV_U8MF2>(src_data, src_step, width, height, minVal, maxVal, minIdx, maxIdx, mask, mask_step);
    case CV_32SC1:
        return minMaxIdxReadOnce<RVV_I32M4, RVV_U8M1, RVV_U32M4>(src_data, src_step, width, height, minVal, maxVal, minIdx, maxIdx, mask, mask_step);
    case CV_32FC1:
        return minMaxIdxReadOnce<RVV_F32M4, RVV_U8M1, RVV_U32M4>(src_data, src_step, width, height, minVal, maxVal, minIdx, maxIdx, mask, mask_step);
    case CV_64FC1:
        return minMaxIdxReadOnce<RVV_F64M4, RVV_U8MF2, RVV_U32M2>(src_data, src_step, width, height, minVal, maxVal, minIdx, maxIdx, mask, mask_step);
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

}}}

#endif
