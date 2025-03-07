// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_MINMAX_HPP_INCLUDED
#define OPENCV_HAL_RVV_MINMAX_HPP_INCLUDED

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv { namespace minmax {

#undef cv_hal_minMaxIdx
#define cv_hal_minMaxIdx cv::cv_hal_rvv::minmax::minMaxIdx
#undef cv_hal_minMaxIdxMaskStep
#define cv_hal_minMaxIdxMaskStep cv::cv_hal_rvv::minmax::minMaxIdx

template<typename T> struct rvv;

#define HAL_RVV_GENERATOR(T, EEW, TYPE, IS_U, EMUL, M_EMUL, B_LEN) \
template<> struct rvv<T> \
{ \
    using vec_t = v##IS_U##int##EEW##EMUL##_t; \
    using bool_t = vbool##B_LEN##_t; \
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e##EEW##EMUL(); } \
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e##EEW##EMUL(a); } \
    static inline vec_t vmv_v_x(T a, size_t b) { return __riscv_vmv_v_x_##TYPE##EMUL(a, b); } \
    static inline vec_t vle(const T* a, size_t b) { return __riscv_vle##EEW##_v_##TYPE##EMUL(a, b); } \
    static inline vuint8##M_EMUL##_t vle_mask(const uchar* a, size_t b) { return __riscv_vle8_v_u8##M_EMUL(a, b); } \
    static inline vec_t vmin_tu(vec_t a, vec_t b, vec_t c, size_t d) { return __riscv_vmin##IS_U##_tu(a, b, c, d); } \
    static inline vec_t vmax_tu(vec_t a, vec_t b, vec_t c, size_t d) { return __riscv_vmax##IS_U##_tu(a, b, c, d); } \
    static inline vec_t vmin_tumu(bool_t a, vec_t b, vec_t c, vec_t d, size_t e) { return __riscv_vmin##IS_U##_tumu(a, b, c, d, e); } \
    static inline vec_t vmax_tumu(bool_t a, vec_t b, vec_t c, vec_t d, size_t e) { return __riscv_vmax##IS_U##_tumu(a, b, c, d, e); } \
    static inline vec_t vredmin(vec_t a, vec_t b, size_t c) { return __riscv_vredmin##IS_U(a, b, c); } \
    static inline vec_t vredmax(vec_t a, vec_t b, size_t c) { return __riscv_vredmax##IS_U(a, b, c); } \
};
HAL_RVV_GENERATOR(uchar , 8 , u8 , u, m1, m1 , 8 )
HAL_RVV_GENERATOR(schar , 8 , i8 ,  , m1, m1 , 8 )
HAL_RVV_GENERATOR(ushort, 16, u16, u, m1, mf2, 16)
HAL_RVV_GENERATOR(short , 16, i16,  , m1, mf2, 16)
#undef HAL_RVV_GENERATOR

#define HAL_RVV_GENERATOR(T, NAME, EEW, TYPE, IS_F, F_OR_S, F_OR_X, EMUL, M_EMUL, P_EMUL, B_LEN) \
template<> struct rvv<T> \
{ \
    using vec_t = v##NAME##EEW##EMUL##_t; \
    using bool_t = vbool##B_LEN##_t; \
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e##EEW##EMUL(); } \
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e##EEW##EMUL(a); } \
    static inline vec_t vmv_v_x(T a, size_t b) { return __riscv_v##IS_F##mv_v_##F_OR_X##_##TYPE##EMUL(a, b); } \
    static inline vuint32##P_EMUL##_t vid(size_t a) { return __riscv_vid_v_u32##P_EMUL(a); } \
    static inline vuint32##P_EMUL##_t vundefined() { return __riscv_vundefined_u32##P_EMUL(); } \
    static inline vec_t vle(const T* a, size_t b) { return __riscv_vle##EEW##_v_##TYPE##EMUL(a, b); } \
    static inline vuint8##M_EMUL##_t vle_mask(const uchar* a, size_t b) { return __riscv_vle8_v_u8##M_EMUL(a, b); } \
    static inline bool_t vmlt(vec_t a, vec_t b, size_t c) { return __riscv_vm##F_OR_S##lt(a, b, c); } \
    static inline bool_t vmgt(vec_t a, vec_t b, size_t c) { return __riscv_vm##F_OR_S##gt(a, b, c); } \
    static inline bool_t vmlt_mu(bool_t a, bool_t b, vec_t c, vec_t d, size_t e) { return __riscv_vm##F_OR_S##lt##_mu(a, b, c, d, e); } \
    static inline bool_t vmgt_mu(bool_t a, bool_t b, vec_t c, vec_t d, size_t e) { return __riscv_vm##F_OR_S##gt##_mu(a, b, c, d, e); } \
    static inline T vmv_x_s(vec_t a) { return __riscv_v##IS_F##mv_##F_OR_X(a); } \
};
HAL_RVV_GENERATOR(int   , int  , 32, i32,  , s, x, m4, m1 , m4, 8 )
HAL_RVV_GENERATOR(float , float, 32, f32, f, f, f, m4, m1 , m4, 8 )
HAL_RVV_GENERATOR(double, float, 64, f64, f, f, f, m4, mf2, m2, 16)
#undef HAL_RVV_GENERATOR

template<typename T>
inline int minMaxIdxReadTwice(const uchar* src_data, size_t src_step, int width, int height, double* minVal, double* maxVal,
                              int* minIdx, int* maxIdx, uchar* mask, size_t mask_step)
{
    int vlmax = rvv<T>::vsetvlmax();
    auto vec_min = rvv<T>::vmv_v_x(std::numeric_limits<T>::max(), vlmax);
    auto vec_max = rvv<T>::vmv_v_x(std::numeric_limits<T>::lowest(), vlmax);
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
                vl = rvv<T>::vsetvl(width - j);
                auto vec_src = rvv<T>::vle(src_row + j, vl);
                auto vec_mask = rvv<T>::vle_mask(mask_row + j, vl);
                auto bool_mask = __riscv_vmsne(vec_mask, 0, vl);
                vec_min = rvv<T>::vmin_tumu(bool_mask, vec_min, vec_min, vec_src, vl);
                vec_max = rvv<T>::vmax_tumu(bool_mask, vec_max, vec_max, vec_src, vl);
            }
        }

        auto sc_minval = rvv<T>::vmv_v_x(std::numeric_limits<T>::max(), vlmax);
        auto sc_maxval = rvv<T>::vmv_v_x(std::numeric_limits<T>::lowest(), vlmax);
        sc_minval = rvv<T>::vredmin(vec_min, sc_minval, vlmax);
        sc_maxval = rvv<T>::vredmax(vec_max, sc_maxval, vlmax);
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
                vl = rvv<T>::vsetvl(width - j);
                auto vec_src = rvv<T>::vle(src_row + j, vl);
                auto vec_mask = rvv<T>::vle_mask(mask_row + j, vl);
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
                vl = rvv<T>::vsetvl(width - j);
                auto vec_src = rvv<T>::vle(src_row + j, vl);
                vec_min = rvv<T>::vmin_tu(vec_min, vec_min, vec_src, vl);
                vec_max = rvv<T>::vmax_tu(vec_max, vec_max, vec_src, vl);
            }
        }

        auto sc_minval = rvv<T>::vmv_v_x(std::numeric_limits<T>::max(), vlmax);
        auto sc_maxval = rvv<T>::vmv_v_x(std::numeric_limits<T>::lowest(), vlmax);
        sc_minval = rvv<T>::vredmin(vec_min, sc_minval, vlmax);
        sc_maxval = rvv<T>::vredmax(vec_max, sc_maxval, vlmax);
        val_min = __riscv_vmv_x(sc_minval);
        val_max = __riscv_vmv_x(sc_maxval);

        bool found_min = !minIdx, found_max = !maxIdx;
        for (int i = 0; i < height && (!found_min || !found_max); i++)
        {
            const T* src_row = reinterpret_cast<const T*>(src_data + i * src_step);
            int vl;
            for (int j = 0; j < width && (!found_min || !found_max); j += vl)
            {
                vl = rvv<T>::vsetvl(width - j);
                auto vec_src = rvv<T>::vle(src_row + j, vl);
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

template<typename T>
inline int minMaxIdxReadOnce(const uchar* src_data, size_t src_step, int width, int height, double* minVal, double* maxVal,
                             int* minIdx, int* maxIdx, uchar* mask, size_t mask_step)
{
    int vlmax = rvv<T>::vsetvlmax();
    auto vec_min = rvv<T>::vmv_v_x(std::numeric_limits<T>::max(), vlmax);
    auto vec_max = rvv<T>::vmv_v_x(std::numeric_limits<T>::lowest(), vlmax);
    auto vec_pos = rvv<T>::vid(vlmax);
    auto vec_minpos = rvv<T>::vundefined(), vec_maxpos = rvv<T>::vundefined();
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
                vl = rvv<T>::vsetvl(width - j);
                auto vec_src = rvv<T>::vle(src_row + j, vl);
                auto vec_mask = rvv<T>::vle_mask(mask_row + j, vl);
                auto bool_mask = __riscv_vmsne(vec_mask, 0, vl);
                auto bool_zero = __riscv_vmxor(bool_mask, bool_mask, vl);

                auto bool_minpos = rvv<T>::vmlt_mu(bool_mask, bool_zero, vec_src, vec_min, vl);
                auto bool_maxpos = rvv<T>::vmgt_mu(bool_mask, bool_zero, vec_src, vec_max, vl);
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
                vl = rvv<T>::vsetvl(width - j);
                auto vec_src = rvv<T>::vle(src_row + j, vl);

                auto bool_minpos = rvv<T>::vmlt(vec_src, vec_min, vl);
                auto bool_maxpos = rvv<T>::vmgt(vec_src, vec_max, vl);
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
        if (val_min > rvv<T>::vmv_x_s(vec_min))
        {
            val_min = rvv<T>::vmv_x_s(vec_min);
            if (minIdx)
            {
                minIdx[0] = __riscv_vmv_x(vec_minpos) / width;
                minIdx[1] = __riscv_vmv_x(vec_minpos) % width;
            }
        }
        if (val_max < rvv<T>::vmv_x_s(vec_max))
        {
            val_max = rvv<T>::vmv_x_s(vec_max);
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
        return minMaxIdxReadTwice<uchar>(src_data, src_step, width, height, minVal, maxVal, minIdx, maxIdx, mask, mask_step);
    case CV_8SC1:
        return minMaxIdxReadTwice<schar>(src_data, src_step, width, height, minVal, maxVal, minIdx, maxIdx, mask, mask_step);
    case CV_16UC1:
        return minMaxIdxReadTwice<ushort>(src_data, src_step, width, height, minVal, maxVal, minIdx, maxIdx, mask, mask_step);
    case CV_16SC1:
        return minMaxIdxReadTwice<short>(src_data, src_step, width, height, minVal, maxVal, minIdx, maxIdx, mask, mask_step);
    case CV_32SC1:
        return minMaxIdxReadOnce<int>(src_data, src_step, width, height, minVal, maxVal, minIdx, maxIdx, mask, mask_step);
    case CV_32FC1:
        return minMaxIdxReadOnce<float>(src_data, src_step, width, height, minVal, maxVal, minIdx, maxIdx, mask, mask_step);
    case CV_64FC1:
        return minMaxIdxReadOnce<double>(src_data, src_step, width, height, minVal, maxVal, minIdx, maxIdx, mask, mask_step);
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

}}}

#endif
