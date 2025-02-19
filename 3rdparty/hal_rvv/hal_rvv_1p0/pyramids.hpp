// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_PYRAMIDS_HPP_INCLUDED
#define OPENCV_HAL_RVV_PYRAMIDS_HPP_INCLUDED

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_pyrdown
#define cv_hal_pyrdown cv::cv_hal_rvv::pyrDown
#undef cv_hal_pyrup
#define cv_hal_pyrup cv::cv_hal_rvv::pyrUp

namespace pyramids {

template<typename T> struct rvv;

template<> struct rvv<uchar>
{
    static inline size_t vsetvl_WT(size_t a) { return __riscv_vsetvl_e32m4(a); }
    static inline vuint8m1_t vle_T(const uchar* a, size_t b) { return __riscv_vle8_v_u8m1(a, b); }
    static inline vint32m4_t vle_WT(const int* a, size_t b) { return __riscv_vle32_v_i32m4(a, b); }
    static inline vuint32m4_t vle_M(const uint* a, size_t b) { return __riscv_vle32_v_u32m4(a, b); }
    static inline vuint8m1_t vlse_T(const uchar* a, ptrdiff_t b, size_t c) { return __riscv_vlse8_v_u8m1(a, b, c); }
    static inline vuint8m1_t vloxei_T(const uchar* a, vuint32m4_t b, size_t c) { return __riscv_vloxei32_v_u8m1(a, b, c); }
    static inline void vse_T(uchar* a, vuint8m1_t b, size_t c) { return __riscv_vse8(a, b, c); }
    static inline vint32m4_t vcvt_T_WT(vuint8m1_t a, size_t b) { return __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4(a, b)); }
    static inline vuint8m1_t vcvt_WT_T(vint32m4_t a, int b, size_t c) { return __riscv_vncvt_x(__riscv_vncvt_x(__riscv_vreinterpret_v_i32m4_u32m4(__riscv_vsra(__riscv_vadd(a, 1 << (b - 1), c), b, c)), c), c); }
};

template<> struct rvv<short>
{
    static inline size_t vsetvl_WT(size_t a) { return __riscv_vsetvl_e32m4(a); }
    static inline vint16m2_t vle_T(const short* a, size_t b) { return __riscv_vle16_v_i16m2(a, b); }
    static inline vint32m4_t vle_WT(const int* a, size_t b) { return __riscv_vle32_v_i32m4(a, b); }
    static inline vuint32m4_t vle_M(const uint* a, size_t b) { return __riscv_vle32_v_u32m4(a, b); }
    static inline vint16m2_t vlse_T(const short* a, ptrdiff_t b, size_t c) { return __riscv_vlse16_v_i16m2(a, b, c); }
    static inline vint16m2_t vloxei_T(const short* a, vuint32m4_t b, size_t c) { return __riscv_vloxei32_v_i16m2(a, b, c); }
    static inline void vse_T(short* a, vint16m2_t b, size_t c) { return __riscv_vse16(a, b, c); }
    static inline vint32m4_t vcvt_T_WT(vint16m2_t a, size_t b) { return __riscv_vsext_vf2(a, b); }
    static inline vint16m2_t vcvt_WT_T(vint32m4_t a, int b, size_t c) { return __riscv_vncvt_x(__riscv_vsra(__riscv_vadd(a, 1 << (b - 1), c), b, c), c); }
};

template<> struct rvv<float>
{
    static inline size_t vsetvl_WT(size_t a) { return __riscv_vsetvl_e32m4(a); }
    static inline vfloat32m4_t vle_T(const float* a, size_t b) { return __riscv_vle32_v_f32m4(a, b); }
    static inline vfloat32m4_t vle_WT(const float* a, size_t b) { return __riscv_vle32_v_f32m4(a, b); }
    static inline vuint32m4_t vle_M(const uint* a, size_t b) { return __riscv_vle32_v_u32m4(a, b); }
    static inline vfloat32m4_t vlse_T(const float* a, ptrdiff_t b, size_t c) { return __riscv_vlse32_v_f32m4(a, b, c); }
    static inline vfloat32m4_t vloxei_T(const float* a, vuint32m4_t b, size_t c) { return __riscv_vloxei32_v_f32m4(a, b, c); }
    static inline void vse_T(float* a, vfloat32m4_t b, size_t c) { return __riscv_vse32(a, b, c); }
};

template<typename T, typename WT> struct pyrDownVec0
{
    void operator()(const T* src, WT* row, const uint* tabM, int start, int end)
    {
        int vl;
        switch (start)
        {
        case 1:
            for( int x = start; x < end; x += vl )
            {
                vl = rvv<T>::vsetvl_WT(end - x);
                auto vec_src0 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + x * 2 - 2, 2 * sizeof(T), vl), vl);
                auto vec_src1 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + x * 2 - 1, 2 * sizeof(T), vl), vl);
                auto vec_src2 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + x * 2, 2 * sizeof(T), vl), vl);
                auto vec_src3 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + x * 2 + 1, 2 * sizeof(T), vl), vl);
                auto vec_src4 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + x * 2 + 2, 2 * sizeof(T), vl), vl);
                __riscv_vse32(row + x, __riscv_vadd(__riscv_vadd(__riscv_vadd(vec_src0, vec_src4, vl), __riscv_vadd(vec_src2, vec_src2, vl), vl),
                                                    __riscv_vsll(__riscv_vadd(__riscv_vadd(vec_src1, vec_src2, vl), vec_src3, vl), 2, vl), vl), vl);
            }
            break;
        case 2:
            for( int x = start / 2; x < end / 2; x += vl )
            {
                vl = rvv<T>::vsetvl_WT(end / 2 - x);
                auto vec_src0 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 - 2) * 2, 4 * sizeof(T), vl), vl);
                auto vec_src1 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 - 1) * 2, 4 * sizeof(T), vl), vl);
                auto vec_src2 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2) * 2, 4 * sizeof(T), vl), vl);
                auto vec_src3 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 + 1) * 2, 4 * sizeof(T), vl), vl);
                auto vec_src4 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 + 2) * 2, 4 * sizeof(T), vl), vl);
                __riscv_vsse32(row + x * 2, 2 * sizeof(WT), __riscv_vadd(__riscv_vadd(__riscv_vadd(vec_src0, vec_src4, vl), __riscv_vadd(vec_src2, vec_src2, vl), vl),
                                                                         __riscv_vsll(__riscv_vadd(__riscv_vadd(vec_src1, vec_src2, vl), vec_src3, vl), 2, vl), vl), vl);
                vec_src0 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 - 2) * 2 + 1, 4 * sizeof(T), vl), vl);
                vec_src1 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 - 1) * 2 + 1, 4 * sizeof(T), vl), vl);
                vec_src2 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2) * 2 + 1, 4 * sizeof(T), vl), vl);
                vec_src3 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 + 1) * 2 + 1, 4 * sizeof(T), vl), vl);
                vec_src4 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 + 2) * 2 + 1, 4 * sizeof(T), vl), vl);
                __riscv_vsse32(row + x * 2 + 1, 2 * sizeof(WT), __riscv_vadd(__riscv_vadd(__riscv_vadd(vec_src0, vec_src4, vl), __riscv_vadd(vec_src2, vec_src2, vl), vl),
                                                                             __riscv_vsll(__riscv_vadd(__riscv_vadd(vec_src1, vec_src2, vl), vec_src3, vl), 2, vl), vl), vl);
            }
            break;
        case 3:
            for( int x = start / 3; x < end / 3; x += vl )
            {
                vl = rvv<T>::vsetvl_WT(end / 3 - x);
                auto vec_src0 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 - 2) * 3, 6 * sizeof(T), vl), vl);
                auto vec_src1 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 - 1) * 3, 6 * sizeof(T), vl), vl);
                auto vec_src2 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2) * 3, 6 * sizeof(T), vl), vl);
                auto vec_src3 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 + 1) * 3, 6 * sizeof(T), vl), vl);
                auto vec_src4 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 + 2) * 3, 6 * sizeof(T), vl), vl);
                __riscv_vsse32(row + x * 3, 3 * sizeof(WT), __riscv_vadd(__riscv_vadd(__riscv_vadd(vec_src0, vec_src4, vl), __riscv_vadd(vec_src2, vec_src2, vl), vl),
                                                                         __riscv_vsll(__riscv_vadd(__riscv_vadd(vec_src1, vec_src2, vl), vec_src3, vl), 2, vl), vl), vl);
                vec_src0 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 - 2) * 3 + 1, 6 * sizeof(T), vl), vl);
                vec_src1 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 - 1) * 3 + 1, 6 * sizeof(T), vl), vl);
                vec_src2 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2) * 3 + 1, 6 * sizeof(T), vl), vl);
                vec_src3 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 + 1) * 3 + 1, 6 * sizeof(T), vl), vl);
                vec_src4 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 + 2) * 3 + 1, 6 * sizeof(T), vl), vl);
                __riscv_vsse32(row + x * 3 + 1, 3 * sizeof(WT), __riscv_vadd(__riscv_vadd(__riscv_vadd(vec_src0, vec_src4, vl), __riscv_vadd(vec_src2, vec_src2, vl), vl),
                                                                             __riscv_vsll(__riscv_vadd(__riscv_vadd(vec_src1, vec_src2, vl), vec_src3, vl), 2, vl), vl), vl);
                vec_src0 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 - 2) * 3 + 2, 6 * sizeof(T), vl), vl);
                vec_src1 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 - 1) * 3 + 2, 6 * sizeof(T), vl), vl);
                vec_src2 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2) * 3 + 2, 6 * sizeof(T), vl), vl);
                vec_src3 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 + 1) * 3 + 2, 6 * sizeof(T), vl), vl);
                vec_src4 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 + 2) * 3 + 2, 6 * sizeof(T), vl), vl);
                __riscv_vsse32(row + x * 3 + 2, 3 * sizeof(WT), __riscv_vadd(__riscv_vadd(__riscv_vadd(vec_src0, vec_src4, vl), __riscv_vadd(vec_src2, vec_src2, vl), vl),
                                                                             __riscv_vsll(__riscv_vadd(__riscv_vadd(vec_src1, vec_src2, vl), vec_src3, vl), 2, vl), vl), vl);
            }
            break;
        case 4:
            for( int x = start / 4; x < end / 4; x += vl )
            {
                vl = rvv<T>::vsetvl_WT(end / 4 - x);
                auto vec_src0 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 - 2) * 4, 8 * sizeof(T), vl), vl);
                auto vec_src1 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 - 1) * 4, 8 * sizeof(T), vl), vl);
                auto vec_src2 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2) * 4, 8 * sizeof(T), vl), vl);
                auto vec_src3 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 + 1) * 4, 8 * sizeof(T), vl), vl);
                auto vec_src4 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 + 2) * 4, 8 * sizeof(T), vl), vl);
                __riscv_vsse32(row + x * 4, 4 * sizeof(WT), __riscv_vadd(__riscv_vadd(__riscv_vadd(vec_src0, vec_src4, vl), __riscv_vadd(vec_src2, vec_src2, vl), vl),
                                                                         __riscv_vsll(__riscv_vadd(__riscv_vadd(vec_src1, vec_src2, vl), vec_src3, vl), 2, vl), vl), vl);
                vec_src0 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 - 2) * 4 + 1, 8 * sizeof(T), vl), vl);
                vec_src1 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 - 1) * 4 + 1, 8 * sizeof(T), vl), vl);
                vec_src2 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2) * 4 + 1, 8 * sizeof(T), vl), vl);
                vec_src3 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 + 1) * 4 + 1, 8 * sizeof(T), vl), vl);
                vec_src4 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 + 2) * 4 + 1, 8 * sizeof(T), vl), vl);
                __riscv_vsse32(row + x * 4 + 1, 4 * sizeof(WT), __riscv_vadd(__riscv_vadd(__riscv_vadd(vec_src0, vec_src4, vl), __riscv_vadd(vec_src2, vec_src2, vl), vl),
                                                                             __riscv_vsll(__riscv_vadd(__riscv_vadd(vec_src1, vec_src2, vl), vec_src3, vl), 2, vl), vl), vl);
                vec_src0 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 - 2) * 4 + 2, 8 * sizeof(T), vl), vl);
                vec_src1 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 - 1) * 4 + 2, 8 * sizeof(T), vl), vl);
                vec_src2 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2) * 4 + 2, 8 * sizeof(T), vl), vl);
                vec_src3 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 + 1) * 4 + 2, 8 * sizeof(T), vl), vl);
                vec_src4 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 + 2) * 4 + 2, 8 * sizeof(T), vl), vl);
                __riscv_vsse32(row + x * 4 + 2, 4 * sizeof(WT), __riscv_vadd(__riscv_vadd(__riscv_vadd(vec_src0, vec_src4, vl), __riscv_vadd(vec_src2, vec_src2, vl), vl),
                                                                             __riscv_vsll(__riscv_vadd(__riscv_vadd(vec_src1, vec_src2, vl), vec_src3, vl), 2, vl), vl), vl);
                vec_src0 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 - 2) * 4 + 3, 8 * sizeof(T), vl), vl);
                vec_src1 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 - 1) * 4 + 3, 8 * sizeof(T), vl), vl);
                vec_src2 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2) * 4 + 3, 8 * sizeof(T), vl), vl);
                vec_src3 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 + 1) * 4 + 3, 8 * sizeof(T), vl), vl);
                vec_src4 = rvv<T>::vcvt_T_WT(rvv<T>::vlse_T(src + (x * 2 + 2) * 4 + 3, 8 * sizeof(T), vl), vl);
                __riscv_vsse32(row + x * 4 + 3, 4 * sizeof(WT), __riscv_vadd(__riscv_vadd(__riscv_vadd(vec_src0, vec_src4, vl), __riscv_vadd(vec_src2, vec_src2, vl), vl),
                                                                             __riscv_vsll(__riscv_vadd(__riscv_vadd(vec_src1, vec_src2, vl), vec_src3, vl), 2, vl), vl), vl);
            }
            break;
        default:
            for( int x = start; x < end; x += vl )
            {
                vl = rvv<T>::vsetvl_WT(end - x);
                auto vec_tabM = rvv<T>::vle_M(tabM + x, vl);
                vec_tabM = __riscv_vmul(__riscv_vsub(vec_tabM, start * 2, vl), sizeof(T), vl);
                auto vec_src0 = rvv<T>::vcvt_T_WT(rvv<T>::vloxei_T(src, vec_tabM, vl), vl);
                vec_tabM =  __riscv_vadd(vec_tabM, start * sizeof(T), vl);
                auto vec_src1 = rvv<T>::vcvt_T_WT(rvv<T>::vloxei_T(src, vec_tabM, vl), vl);
                vec_tabM =  __riscv_vadd(vec_tabM, start * sizeof(T), vl);
                auto vec_src2 = rvv<T>::vcvt_T_WT(rvv<T>::vloxei_T(src, vec_tabM, vl), vl);
                vec_tabM =  __riscv_vadd(vec_tabM, start * sizeof(T), vl);
                auto vec_src3 = rvv<T>::vcvt_T_WT(rvv<T>::vloxei_T(src, vec_tabM, vl), vl);
                vec_tabM =  __riscv_vadd(vec_tabM, start * sizeof(T), vl);
                auto vec_src4 = rvv<T>::vcvt_T_WT(rvv<T>::vloxei_T(src, vec_tabM, vl), vl);
                __riscv_vse32(row + x, __riscv_vadd(__riscv_vadd(__riscv_vadd(vec_src0, vec_src4, vl), __riscv_vadd(vec_src2, vec_src2, vl), vl),
                                                    __riscv_vsll(__riscv_vadd(__riscv_vadd(vec_src1, vec_src2, vl), vec_src3, vl), 2, vl), vl), vl);
            }
        }
    }
};
template<> struct pyrDownVec0<float, float>
{
    void operator()(const float* src, float* row, const uint* tabM, int start, int end)
    {
        int vl;
        switch (start)
        {
        case 1:
            for( int x = start; x < end; x += vl )
            {
                vl = rvv<float>::vsetvl_WT(end - x);
                auto vec_src0 = rvv<float>::vlse_T(src + x * 2 - 2, 2 * sizeof(float), vl);
                auto vec_src1 = rvv<float>::vlse_T(src + x * 2 - 1, 2 * sizeof(float), vl);
                auto vec_src2 = rvv<float>::vlse_T(src + x * 2, 2 * sizeof(float), vl);
                auto vec_src3 = rvv<float>::vlse_T(src + x * 2 + 1, 2 * sizeof(float), vl);
                auto vec_src4 = rvv<float>::vlse_T(src + x * 2 + 2, 2 * sizeof(float), vl);
                __riscv_vse32(row + x, __riscv_vfmadd(vec_src2, 6, __riscv_vfmadd(__riscv_vfadd(vec_src1, vec_src3, vl), 4, __riscv_vfadd(vec_src0, vec_src4, vl), vl), vl), vl);
            }
            break;
        case 2:
            for( int x = start / 2; x < end / 2; x += vl )
            {
                vl = rvv<float>::vsetvl_WT(end / 2 - x);
                auto vec_src0 = rvv<float>::vlse_T(src + (x * 2 - 2) * 2, 4 * sizeof(float), vl);
                auto vec_src1 = rvv<float>::vlse_T(src + (x * 2 - 1) * 2, 4 * sizeof(float), vl);
                auto vec_src2 = rvv<float>::vlse_T(src + (x * 2) * 2, 4 * sizeof(float), vl);
                auto vec_src3 = rvv<float>::vlse_T(src + (x * 2 + 1) * 2, 4 * sizeof(float), vl);
                auto vec_src4 = rvv<float>::vlse_T(src + (x * 2 + 2) * 2, 4 * sizeof(float), vl);
                __riscv_vsse32(row + x * 2, 2 * sizeof(float), __riscv_vfmadd(vec_src2, 6, __riscv_vfmadd(__riscv_vfadd(vec_src1, vec_src3, vl), 4, __riscv_vfadd(vec_src0, vec_src4, vl), vl), vl), vl);
                vec_src0 = rvv<float>::vlse_T(src + (x * 2 - 2) * 2 + 1, 4 * sizeof(float), vl);
                vec_src1 = rvv<float>::vlse_T(src + (x * 2 - 1) * 2 + 1, 4 * sizeof(float), vl);
                vec_src2 = rvv<float>::vlse_T(src + (x * 2) * 2 + 1, 4 * sizeof(float), vl);
                vec_src3 = rvv<float>::vlse_T(src + (x * 2 + 1) * 2 + 1, 4 * sizeof(float), vl);
                vec_src4 = rvv<float>::vlse_T(src + (x * 2 + 2) * 2 + 1, 4 * sizeof(float), vl);
                __riscv_vsse32(row + x * 2 + 1, 2 * sizeof(float), __riscv_vfmadd(vec_src2, 6, __riscv_vfmadd(__riscv_vfadd(vec_src1, vec_src3, vl), 4, __riscv_vfadd(vec_src0, vec_src4, vl), vl), vl), vl);
            }
            break;
        case 3:
            for( int x = start / 3; x < end / 3; x += vl )
            {
                vl = rvv<float>::vsetvl_WT(end / 3 - x);
                auto vec_src0 = rvv<float>::vlse_T(src + (x * 2 - 2) * 3, 6 * sizeof(float), vl);
                auto vec_src1 = rvv<float>::vlse_T(src + (x * 2 - 1) * 3, 6 * sizeof(float), vl);
                auto vec_src2 = rvv<float>::vlse_T(src + (x * 2) * 3, 6 * sizeof(float), vl);
                auto vec_src3 = rvv<float>::vlse_T(src + (x * 2 + 1) * 3, 6 * sizeof(float), vl);
                auto vec_src4 = rvv<float>::vlse_T(src + (x * 2 + 2) * 3, 6 * sizeof(float), vl);
                __riscv_vsse32(row + x * 3, 3 * sizeof(float), __riscv_vfmadd(vec_src2, 6, __riscv_vfmadd(__riscv_vfadd(vec_src1, vec_src3, vl), 4, __riscv_vfadd(vec_src0, vec_src4, vl), vl), vl), vl);
                vec_src0 = rvv<float>::vlse_T(src + (x * 2 - 2) * 3 + 1, 6 * sizeof(float), vl);
                vec_src1 = rvv<float>::vlse_T(src + (x * 2 - 1) * 3 + 1, 6 * sizeof(float), vl);
                vec_src2 = rvv<float>::vlse_T(src + (x * 2) * 3 + 1, 6 * sizeof(float), vl);
                vec_src3 = rvv<float>::vlse_T(src + (x * 2 + 1) * 3 + 1, 6 * sizeof(float), vl);
                vec_src4 = rvv<float>::vlse_T(src + (x * 2 + 2) * 3 + 1, 6 * sizeof(float), vl);
                __riscv_vsse32(row + x * 3 + 1, 3 * sizeof(float), __riscv_vfmadd(vec_src2, 6, __riscv_vfmadd(__riscv_vfadd(vec_src1, vec_src3, vl), 4, __riscv_vfadd(vec_src0, vec_src4, vl), vl), vl), vl);
                vec_src0 = rvv<float>::vlse_T(src + (x * 2 - 2) * 3 + 2, 6 * sizeof(float), vl);
                vec_src1 = rvv<float>::vlse_T(src + (x * 2 - 1) * 3 + 2, 6 * sizeof(float), vl);
                vec_src2 = rvv<float>::vlse_T(src + (x * 2) * 3 + 2, 6 * sizeof(float), vl);
                vec_src3 = rvv<float>::vlse_T(src + (x * 2 + 1) * 3 + 2, 6 * sizeof(float), vl);
                vec_src4 = rvv<float>::vlse_T(src + (x * 2 + 2) * 3 + 2, 6 * sizeof(float), vl);
                __riscv_vsse32(row + x * 3 + 2, 3 * sizeof(float), __riscv_vfmadd(vec_src2, 6, __riscv_vfmadd(__riscv_vfadd(vec_src1, vec_src3, vl), 4, __riscv_vfadd(vec_src0, vec_src4, vl), vl), vl), vl);
            }
            break;
        case 4:
            for( int x = start / 4; x < end / 4; x += vl )
            {
                vl = rvv<float>::vsetvl_WT(end / 4 - x);
                auto vec_src0 = rvv<float>::vlse_T(src + (x * 2 - 2) * 4, 8 * sizeof(float), vl);
                auto vec_src1 = rvv<float>::vlse_T(src + (x * 2 - 1) * 4, 8 * sizeof(float), vl);
                auto vec_src2 = rvv<float>::vlse_T(src + (x * 2) * 4, 8 * sizeof(float), vl);
                auto vec_src3 = rvv<float>::vlse_T(src + (x * 2 + 1) * 4, 8 * sizeof(float), vl);
                auto vec_src4 = rvv<float>::vlse_T(src + (x * 2 + 2) * 4, 8 * sizeof(float), vl);
                __riscv_vsse32(row + x * 4, 4 * sizeof(float), __riscv_vfmadd(vec_src2, 6, __riscv_vfmadd(__riscv_vfadd(vec_src1, vec_src3, vl), 4, __riscv_vfadd(vec_src0, vec_src4, vl), vl), vl), vl);
                vec_src0 = rvv<float>::vlse_T(src + (x * 2 - 2) * 4 + 1, 8 * sizeof(float), vl);
                vec_src1 = rvv<float>::vlse_T(src + (x * 2 - 1) * 4 + 1, 8 * sizeof(float), vl);
                vec_src2 = rvv<float>::vlse_T(src + (x * 2) * 4 + 1, 8 * sizeof(float), vl);
                vec_src3 = rvv<float>::vlse_T(src + (x * 2 + 1) * 4 + 1, 8 * sizeof(float), vl);
                vec_src4 = rvv<float>::vlse_T(src + (x * 2 + 2) * 4 + 1, 8 * sizeof(float), vl);
                __riscv_vsse32(row + x * 4 + 1, 4 * sizeof(float), __riscv_vfmadd(vec_src2, 6, __riscv_vfmadd(__riscv_vfadd(vec_src1, vec_src3, vl), 4, __riscv_vfadd(vec_src0, vec_src4, vl), vl), vl), vl);
                vec_src0 = rvv<float>::vlse_T(src + (x * 2 - 2) * 4 + 2, 8 * sizeof(float), vl);
                vec_src1 = rvv<float>::vlse_T(src + (x * 2 - 1) * 4 + 2, 8 * sizeof(float), vl);
                vec_src2 = rvv<float>::vlse_T(src + (x * 2) * 4 + 2, 8 * sizeof(float), vl);
                vec_src3 = rvv<float>::vlse_T(src + (x * 2 + 1) * 4 + 2, 8 * sizeof(float), vl);
                vec_src4 = rvv<float>::vlse_T(src + (x * 2 + 2) * 4 + 2, 8 * sizeof(float), vl);
                __riscv_vsse32(row + x * 4 + 2, 4 * sizeof(float), __riscv_vfmadd(vec_src2, 6, __riscv_vfmadd(__riscv_vfadd(vec_src1, vec_src3, vl), 4, __riscv_vfadd(vec_src0, vec_src4, vl), vl), vl), vl);
                vec_src0 = rvv<float>::vlse_T(src + (x * 2 - 2) * 4 + 3, 8 * sizeof(float), vl);
                vec_src1 = rvv<float>::vlse_T(src + (x * 2 - 1) * 4 + 3, 8 * sizeof(float), vl);
                vec_src2 = rvv<float>::vlse_T(src + (x * 2) * 4 + 3, 8 * sizeof(float), vl);
                vec_src3 = rvv<float>::vlse_T(src + (x * 2 + 1) * 4 + 3, 8 * sizeof(float), vl);
                vec_src4 = rvv<float>::vlse_T(src + (x * 2 + 2) * 4 + 3, 8 * sizeof(float), vl);
                __riscv_vsse32(row + x * 4 + 3, 4 * sizeof(float), __riscv_vfmadd(vec_src2, 6, __riscv_vfmadd(__riscv_vfadd(vec_src1, vec_src3, vl), 4, __riscv_vfadd(vec_src0, vec_src4, vl), vl), vl), vl);
            }
            break;
        default:
            for( int x = start; x < end; x += vl )
            {
                vl = rvv<float>::vsetvl_WT(end - x);
                auto vec_tabM = rvv<float>::vle_M(tabM + x, vl);
                vec_tabM = __riscv_vmul(__riscv_vsub(vec_tabM, start * 2, vl), sizeof(float), vl);
                auto vec_src0 = rvv<float>::vloxei_T(src, vec_tabM, vl);
                vec_tabM =  __riscv_vadd(vec_tabM, start * sizeof(float), vl);
                auto vec_src1 = rvv<float>::vloxei_T(src, vec_tabM, vl);
                vec_tabM =  __riscv_vadd(vec_tabM, start * sizeof(float), vl);
                auto vec_src2 = rvv<float>::vloxei_T(src, vec_tabM, vl);
                vec_tabM =  __riscv_vadd(vec_tabM, start * sizeof(float), vl);
                auto vec_src3 = rvv<float>::vloxei_T(src, vec_tabM, vl);
                vec_tabM =  __riscv_vadd(vec_tabM, start * sizeof(float), vl);
                auto vec_src4 = rvv<float>::vloxei_T(src, vec_tabM, vl);
                __riscv_vse32(row + x, __riscv_vfmadd(__riscv_vfadd(__riscv_vfadd(vec_src1, vec_src2, vl), vec_src3, vl), 4,
                                                      __riscv_vfadd(__riscv_vfadd(vec_src0, vec_src4, vl), __riscv_vfadd(vec_src2, vec_src2, vl), vl), vl), vl);
            }
        }
    }
};

template<typename T, typename WT> struct pyrDownVec1
{
    void operator()(WT* row0, WT* row1, WT* row2, WT* row3, WT* row4, T* dst, int end)
    {
        int vl;
        for( int x = 0 ; x < end; x += vl )
        {
            vl = pyramids::rvv<T>::vsetvl_WT(end - x);
            auto vec_src0 = pyramids::rvv<T>::vle_WT(row0 + x, vl);
            auto vec_src1 = pyramids::rvv<T>::vle_WT(row1 + x, vl);
            auto vec_src2 = pyramids::rvv<T>::vle_WT(row2 + x, vl);
            auto vec_src3 = pyramids::rvv<T>::vle_WT(row3 + x, vl);
            auto vec_src4 = pyramids::rvv<T>::vle_WT(row4 + x, vl);
            pyramids::rvv<T>::vse_T(dst + x, pyramids::rvv<T>::vcvt_WT_T(__riscv_vadd(__riscv_vadd(__riscv_vadd(vec_src0, vec_src4, vl), __riscv_vadd(vec_src2, vec_src2, vl), vl),
                                                                                      __riscv_vsll(__riscv_vadd(__riscv_vadd(vec_src1, vec_src2, vl), vec_src3, vl), 2, vl), vl), 8, vl), vl);
        }
    }
};
template<> struct pyrDownVec1<float, float>
{
    void operator()(float* row0, float* row1, float* row2, float* row3, float* row4, float* dst, int end)
    {
        int vl;
        for( int x = 0 ; x < end; x += vl )
        {
            vl = pyramids::rvv<float>::vsetvl_WT(end - x);
            auto vec_src0 = pyramids::rvv<float>::vle_WT(row0 + x, vl);
            auto vec_src1 = pyramids::rvv<float>::vle_WT(row1 + x, vl);
            auto vec_src2 = pyramids::rvv<float>::vle_WT(row2 + x, vl);
            auto vec_src3 = pyramids::rvv<float>::vle_WT(row3 + x, vl);
            auto vec_src4 = pyramids::rvv<float>::vle_WT(row4 + x, vl);
            pyramids::rvv<float>::vse_T(dst + x, __riscv_vfmul(__riscv_vfmadd(vec_src2, 6, __riscv_vfmadd(__riscv_vfadd(vec_src1, vec_src3, vl), 4, __riscv_vfadd(vec_src0, vec_src4, vl), vl), vl), 1.f / 256.f, vl), vl);
        }
    }
};

template<typename T, typename WT> struct pyrUpVec0
{
    void operator()(const T* src, WT* row, const uint* dtab, int start, int end)
    {
        int vl;
        for( int x = start; x < end; x += vl )
        {
            vl = rvv<T>::vsetvl_WT(end - x);
            auto vec_src0 = rvv<T>::vcvt_T_WT(rvv<T>::vle_T(src + x - start, vl), vl);
            auto vec_src1 = rvv<T>::vcvt_T_WT(rvv<T>::vle_T(src + x, vl), vl);
            auto vec_src2 = rvv<T>::vcvt_T_WT(rvv<T>::vle_T(src + x + start, vl), vl);

            auto vec_dtab = rvv<T>::vle_M(dtab + x, vl);
            vec_dtab = __riscv_vmul(vec_dtab, sizeof(WT), vl);
            __riscv_vsoxei32(row, vec_dtab, __riscv_vadd(__riscv_vadd(vec_src0, vec_src2, vl), __riscv_vadd(__riscv_vsll(vec_src1, 2, vl), __riscv_vsll(vec_src1, 1, vl), vl), vl), vl);
            __riscv_vsoxei32(row, __riscv_vadd(vec_dtab, start * sizeof(WT), vl), __riscv_vsll(__riscv_vadd(vec_src1, vec_src2, vl), 2, vl), vl);
        }
    }
};
template<> struct pyrUpVec0<float, float>
{
    void operator()(const float* src, float* row, const uint* dtab, int start, int end)
    {
        int vl;
        for( int x = start; x < end; x += vl )
        {
            vl = rvv<float>::vsetvl_WT(end - x);
            auto vec_src0 = rvv<float>::vle_T(src + x - start, vl);
            auto vec_src1 = rvv<float>::vle_T(src + x, vl);
            auto vec_src2 = rvv<float>::vle_T(src + x + start, vl);

            auto vec_dtab = rvv<float>::vle_M(dtab + x, vl);
            vec_dtab = __riscv_vmul(vec_dtab, sizeof(float), vl);
            __riscv_vsoxei32(row, vec_dtab, __riscv_vfadd(__riscv_vfmadd(vec_src1, 6, vec_src0, vl), vec_src2, vl), vl);
            __riscv_vsoxei32(row, __riscv_vadd(vec_dtab, start * sizeof(float), vl), __riscv_vfmul(__riscv_vfadd(vec_src1, vec_src2, vl), 4, vl), vl);
        }
    }
};

template<typename T, typename WT> struct pyrUpVec1
{
    void operator()(WT* row0, WT* row1, WT* row2, T* dst0, T* dst1, int end)
    {
        int vl;
        if (dst0 != dst1)
        {
            for( int x = 0 ; x < end; x += vl )
            {
                vl = pyramids::rvv<T>::vsetvl_WT(end - x);
                auto vec_src0 = pyramids::rvv<T>::vle_WT(row0 + x, vl);
                auto vec_src1 = pyramids::rvv<T>::vle_WT(row1 + x, vl);
                auto vec_src2 = pyramids::rvv<T>::vle_WT(row2 + x, vl);
                pyramids::rvv<T>::vse_T(dst0 + x, pyramids::rvv<T>::vcvt_WT_T(__riscv_vadd(__riscv_vadd(vec_src0, vec_src2, vl), __riscv_vadd(__riscv_vsll(vec_src1, 2, vl), __riscv_vsll(vec_src1, 1, vl), vl), vl), 6, vl), vl);
                pyramids::rvv<T>::vse_T(dst1 + x, pyramids::rvv<T>::vcvt_WT_T(__riscv_vsll(__riscv_vadd(vec_src1, vec_src2, vl), 2, vl), 6, vl), vl);
            }
        }
        else
        {
            for( int x = 0 ; x < end; x += vl )
            {
                vl = pyramids::rvv<T>::vsetvl_WT(end - x);
                auto vec_src0 = pyramids::rvv<T>::vle_WT(row0 + x, vl);
                auto vec_src1 = pyramids::rvv<T>::vle_WT(row1 + x, vl);
                auto vec_src2 = pyramids::rvv<T>::vle_WT(row2 + x, vl);
                pyramids::rvv<T>::vse_T(dst0 + x, pyramids::rvv<T>::vcvt_WT_T(__riscv_vadd(__riscv_vadd(vec_src0, vec_src2, vl), __riscv_vadd(__riscv_vsll(vec_src1, 2, vl), __riscv_vsll(vec_src1, 1, vl), vl), vl), 6, vl), vl);
            }
        }
    }
};
template<> struct pyrUpVec1<float, float>
{
    void operator()(float* row0, float* row1, float* row2, float* dst0, float* dst1, int end)
    {
        int vl;
        if (dst0 != dst1)
        {
            for( int x = 0 ; x < end; x += vl )
            {
                vl = pyramids::rvv<float>::vsetvl_WT(end - x);
                auto vec_src0 = pyramids::rvv<float>::vle_WT(row0 + x, vl);
                auto vec_src1 = pyramids::rvv<float>::vle_WT(row1 + x, vl);
                auto vec_src2 = pyramids::rvv<float>::vle_WT(row2 + x, vl);
                pyramids::rvv<float>::vse_T(dst0 + x, __riscv_vfmul(__riscv_vfadd(__riscv_vfmadd(vec_src1, 6, vec_src0, vl), vec_src2, vl), 1.f / 64.f, vl), vl);
                pyramids::rvv<float>::vse_T(dst1 + x, __riscv_vfmul(__riscv_vfadd(vec_src1, vec_src2, vl), 1.f / 16.f, vl), vl);
            }
        }
        else
        {
            for( int x = 0 ; x < end; x += vl )
            {
                vl = pyramids::rvv<float>::vsetvl_WT(end - x);
                auto vec_src0 = pyramids::rvv<float>::vle_WT(row0 + x, vl);
                auto vec_src1 = pyramids::rvv<float>::vle_WT(row1 + x, vl);
                auto vec_src2 = pyramids::rvv<float>::vle_WT(row2 + x, vl);
                pyramids::rvv<float>::vse_T(dst0 + x, __riscv_vfmul(__riscv_vfadd(__riscv_vfmadd(vec_src1, 6, vec_src0, vl), vec_src2, vl), 1.f / 64.f, vl), vl);
            }
        }
    }
};

} // cv::cv_hal_rvv::pyramids

template<typename T, typename WT>
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

// the algorithm is copied from imgproc/src/pyramids.cpp,
// in the function template void cv::pyrDown_
template<typename T, typename WT>
inline int pyrDown(const uchar* src_data, size_t src_step, int src_width, int src_height, uchar* dst_data, size_t dst_step, int dst_width, int dst_height, int cn, int borderType)
{
    const int PD_SZ = 5;

    AutoBuffer<int> _tabM(dst_width * cn), _tabL(cn * (PD_SZ + 2)),
        _tabR(cn * (PD_SZ + 2));
    int *tabM = _tabM.data(), *tabL = _tabL.data(), *tabR = _tabR.data();

    CV_Assert( src_width > 0 && src_height > 0 &&
               std::abs(dst_width*2 - src_width) <= 2 &&
               std::abs(dst_height*2 - src_height) <= 2 );
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

    cv::parallel_for_(Range(0,dst_height), PyrDownInvoker<T, WT>(src_data, src_step, src_width, src_height, dst_data, dst_step, dst_width, dst_height, cn, borderType, tabR, tabM, tabL), cv::getNumThreads());
    return CV_HAL_ERROR_OK;
}

template<typename T, typename WT>
void PyrDownInvoker<T, WT>::operator()(const Range& range) const
{
    const int PD_SZ = 5;

    int bufstep = (int)alignSize(dst_width*cn, 16);
    AutoBuffer<WT> _buf(bufstep*PD_SZ + 16);
    WT* buf = alignPtr((WT*)_buf.data(), 16);
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

                pyramids::pyrDownVec0<T, WT>()(src, row, reinterpret_cast<const uint*>(tabM), cn, width0);
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

        pyramids::pyrDownVec1<T, WT>()(row0, row1, row2, row3, row4, dst, _dst_width);
    }
}

// the algorithm is copied from imgproc/src/pyramids.cpp,
// in the function template void cv::pyrUp_
template<typename T, typename WT>
inline int pyrUp(const uchar* src_data, size_t src_step, int src_width, int src_height, uchar* dst_data, size_t dst_step, int dst_width, int dst_height, int cn)
{
    const int PU_SZ = 3;

    int bufstep = (int)alignSize((dst_width+1)*cn, 16);
    AutoBuffer<WT> _buf(bufstep*PU_SZ + 16);
    WT* buf = alignPtr((WT*)_buf.data(), 16);
    AutoBuffer<int> _dtab(src_width*cn);
    int* dtab = _dtab.data();
    WT* rows[PU_SZ];

    CV_Assert( std::abs(dst_width - src_width*2) == dst_width % 2 &&
               std::abs(dst_height - src_height*2) == dst_height % 2);
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
            int _sy = borderInterpolate(sy*2, src_height*2, BORDER_REFLECT_101)/2;
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

            pyramids::pyrUpVec0<T, WT>()(src, row, reinterpret_cast<const uint*>(dtab), cn, src_width - cn);
        }

        // do vertical convolution and decimation and write the result to the destination image
        for( k = 0; k < PU_SZ; k++ )
            rows[k] = buf + ((y - PU_SZ/2 + k - sy0) % PU_SZ)*bufstep;
        row0 = rows[0]; row1 = rows[1]; row2 = rows[2];

        pyramids::pyrUpVec1<T, WT>()(row0, row1, row2, dst0, dst1, dst_width);
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
        return pyrDown<uchar, int>(src_data, src_step, src_width, src_height, dst_data, dst_step, dst_width, dst_height, cn, border_type);
    case CV_16S:
        return pyrDown<short, int>(src_data, src_step, src_width, src_height, dst_data, dst_step, dst_width, dst_height, cn, border_type);
    case CV_32F:
        return pyrDown<float, float>(src_data, src_step, src_width, src_height, dst_data, dst_step, dst_width, dst_height, cn, border_type);
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
        return pyrUp<uchar, int>(src_data, src_step, src_width, src_height, dst_data, dst_step, dst_width, dst_height, cn);
    case CV_16S:
        return pyrUp<short, int>(src_data, src_step, src_width, src_height, dst_data, dst_step, dst_width, dst_height, cn);
    case CV_32F:
        return pyrUp<float, float>(src_data, src_step, src_width, src_height, dst_data, dst_step, dst_width, dst_height, cn);
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

}}

#endif
