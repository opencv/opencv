// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_COLOR_HPP_INCLUDED
#define OPENCV_HAL_RVV_COLOR_HPP_INCLUDED

#include <riscv_vector.h>
#include "thread_pool.hpp"

namespace cv { namespace cv_hal_rvv {

namespace BGRtoBGR {
#undef cv_hal_cvtBGRtoBGR
#define cv_hal_cvtBGRtoBGR cv::cv_hal_rvv::BGRtoBGR::cvtBGRtoBGR

template<typename T> struct rvv;
template<> struct rvv<uchar>
{
    using T = vuint8m2_t;
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e8m2(); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e8m2(a); }
    static inline void vlseg(const uchar* a, int b, T& c, T& d, T& e, T& f, size_t g)
    {
        if (b == 3)
        {
            auto x = __riscv_vlseg3e8_v_u8m2x3(a, g);
            c = __riscv_vget_v_u8m2x3_u8m2(x, 0), d = __riscv_vget_v_u8m2x3_u8m2(x, 1), e = __riscv_vget_v_u8m2x3_u8m2(x, 2);
        }
        else
        {
            auto x = __riscv_vlseg4e8_v_u8m2x4(a, g);
            c = __riscv_vget_v_u8m2x4_u8m2(x, 0), d = __riscv_vget_v_u8m2x4_u8m2(x, 1), e = __riscv_vget_v_u8m2x4_u8m2(x, 2), f = __riscv_vget_v_u8m2x4_u8m2(x, 3);
        }
    }
    static inline void vsseg(uchar* a, int b, T c, T d, T e, T f, size_t g) { return b == 3 ? __riscv_vsseg3e8(a, __riscv_vcreate_v_u8m2x3(c, d, e), g) : __riscv_vsseg4e8(a, __riscv_vcreate_v_u8m2x4(c, d, e, f), g); }
    static inline T vmv_v_x(uchar a, size_t b) { return __riscv_vmv_v_x_u8m2(a, b); }
};
template<> struct rvv<ushort>
{
    using T = vuint16m2_t;
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e16m2(); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e16m2(a); }
    static inline void vlseg(const ushort* a, int b, T& c, T& d, T& e, T& f, size_t g)
    {
        if (b == 3)
        {
            auto x = __riscv_vlseg3e16_v_u16m2x3(a, g);
            c = __riscv_vget_v_u16m2x3_u16m2(x, 0), d = __riscv_vget_v_u16m2x3_u16m2(x, 1), e = __riscv_vget_v_u16m2x3_u16m2(x, 2);
        }
        else
        {
            auto x = __riscv_vlseg4e16_v_u16m2x4(a, g);
            c = __riscv_vget_v_u16m2x4_u16m2(x, 0), d = __riscv_vget_v_u16m2x4_u16m2(x, 1), e = __riscv_vget_v_u16m2x4_u16m2(x, 2), f = __riscv_vget_v_u16m2x4_u16m2(x, 3);
        }
    }
    static inline void vsseg(ushort* a, int b, T c, T d, T e, T f, size_t g) { return b == 3 ? __riscv_vsseg3e16(a, __riscv_vcreate_v_u16m2x3(c, d, e), g) : __riscv_vsseg4e16(a, __riscv_vcreate_v_u16m2x4(c, d, e, f), g); }
    static inline T vmv_v_x(ushort a, size_t b) { return __riscv_vmv_v_x_u16m2(a, b); }
};
template<> struct rvv<float>
{
    using T = vfloat32m2_t;
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e32m2(); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e32m2(a); }
    static inline void vlseg(const float* a, int b, T& c, T& d, T& e, T& f, size_t g)
    {
        if (b == 3)
        {
            auto x = __riscv_vlseg3e32_v_f32m2x3(a, g);
            c = __riscv_vget_v_f32m2x3_f32m2(x, 0), d = __riscv_vget_v_f32m2x3_f32m2(x, 1), e = __riscv_vget_v_f32m2x3_f32m2(x, 2);
        }
        else
        {
            auto x = __riscv_vlseg4e32_v_f32m2x4(a, g);
            c = __riscv_vget_v_f32m2x4_f32m2(x, 0), d = __riscv_vget_v_f32m2x4_f32m2(x, 1), e = __riscv_vget_v_f32m2x4_f32m2(x, 2), f = __riscv_vget_v_f32m2x4_f32m2(x, 3);
        }
    }
    static inline void vsseg(float* a, int b, T c, T d, T e, T f, size_t g) { return b == 3 ? __riscv_vsseg3e32(a, __riscv_vcreate_v_f32m2x3(c, d, e), g) : __riscv_vsseg4e32(a, __riscv_vcreate_v_f32m2x4(c, d, e, f), g); }
    static inline T vmv_v_x(float a, size_t b) { return __riscv_vfmv_v_f_f32m2(a, b); }
};

template<typename T>
inline int cvtBGRtoBGR(int start, int end, const T * src, size_t src_step, T * dst, size_t dst_step, int width, int scn, int dcn, bool swapBlue)
{
    if ((scn != 3 && scn != 4) || (dcn != 3 && dcn != 4))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    src_step /= sizeof(T);
    dst_step /= sizeof(T);

    if (scn == dcn && !swapBlue)
    {
        for (int i = start; i < end; i++)
            memcpy(dst + i * dst_step, src + i * src_step, sizeof(T) * width * scn);
    }
    else
    {
        auto alpha = rvv<T>::vmv_v_x(typeid(T) == typeid(float) ? 1.0f : std::numeric_limits<T>::max(), rvv<T>::vsetvlmax());
        int vl;
        for (int i = start; i < end; i++)
        {
            for (int j = 0; j < width; j += vl)
            {
                vl = rvv<T>::vsetvl(width - j);
                typename rvv<T>::T vec_srcB, vec_srcG, vec_srcR, vec_srcA;
                rvv<T>::vlseg(src + i * src_step + j * scn, scn, vec_srcB, vec_srcG, vec_srcR, vec_srcA, vl);
                if (swapBlue)
                {
                    auto t = vec_srcB;
                    vec_srcB = vec_srcR, vec_srcR = t;
                }
                rvv<T>::vsseg(dst + i * dst_step + j * dcn, dcn, vec_srcB, vec_srcG, vec_srcR, scn == 3 && dcn == 4 ? alpha : vec_srcA, vl);
            }
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int cvtBGRtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int depth, int scn, int dcn, bool swapBlue)
{
    switch (depth)
    {
    case CV_8U:
        return ThreadPool::parallel_for(height, -1, cvtBGRtoBGR<uchar>, reinterpret_cast<const uchar*>(src_data), src_step, reinterpret_cast<uchar*>(dst_data), dst_step, width, scn, dcn, swapBlue);
    case CV_16U:
        return ThreadPool::parallel_for(height, -1, cvtBGRtoBGR<ushort>, reinterpret_cast<const ushort*>(src_data), src_step, reinterpret_cast<ushort*>(dst_data), dst_step, width, scn, dcn, swapBlue);
    case CV_32F:
        return ThreadPool::parallel_for(height, -1, cvtBGRtoBGR<float>, reinterpret_cast<const float*>(src_data), src_step, reinterpret_cast<float*>(dst_data), dst_step, width, scn, dcn, swapBlue);
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}
} // cv::cv_hal_rvv::BGRtoBGR

namespace GraytoBGR {
#undef cv_hal_cvtGraytoBGR
#define cv_hal_cvtGraytoBGR cv::cv_hal_rvv::GraytoBGR::cvtGraytoBGR

template<typename T> struct rvv;
template<> struct rvv<uchar>
{
    using T = vuint8m2_t;
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e8m2(); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e8m2(a); }
    static inline T vle(const uchar* a, size_t b) { return __riscv_vle8_v_u8m2(a, b); }
    static inline void vsseg(uchar* a, int b, T c, T d, T e, T f, size_t g) { return b == 3 ? __riscv_vsseg3e8(a, __riscv_vcreate_v_u8m2x3(c, d, e), g) : __riscv_vsseg4e8(a, __riscv_vcreate_v_u8m2x4(c, d, e, f), g); }
    static inline T vmv_v_x(uchar a, size_t b) { return __riscv_vmv_v_x_u8m2(a, b); }
};
template<> struct rvv<ushort>
{
    using T = vuint16m2_t;
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e16m2(); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e16m2(a); }
    static inline T vle(const ushort* a, size_t b) { return __riscv_vle16_v_u16m2(a, b); }
    static inline void vsseg(ushort* a, int b, T c, T d, T e, T f, size_t g) { return b == 3 ? __riscv_vsseg3e16(a, __riscv_vcreate_v_u16m2x3(c, d, e), g) : __riscv_vsseg4e16(a, __riscv_vcreate_v_u16m2x4(c, d, e, f), g); }
    static inline T vmv_v_x(ushort a, size_t b) { return __riscv_vmv_v_x_u16m2(a, b); }
};
template<> struct rvv<float>
{
    using T = vfloat32m2_t;
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e32m2(); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e32m2(a); }
    static inline T vle(const float* a, size_t b) { return __riscv_vle32_v_f32m2(a, b); }
    static inline void vsseg(float* a, int b, T c, T d, T e, T f, size_t g) { return b == 3 ? __riscv_vsseg3e32(a, __riscv_vcreate_v_f32m2x3(c, d, e), g) : __riscv_vsseg4e32(a, __riscv_vcreate_v_f32m2x4(c, d, e, f), g); }
    static inline T vmv_v_x(float a, size_t b) { return __riscv_vfmv_v_f_f32m2(a, b); }
};

template<typename T>
inline int cvtGraytoBGR(int start, int end, const T * src, size_t src_step, T * dst, size_t dst_step, int width, int dcn)
{
    if (dcn != 3 && dcn != 4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    src_step /= sizeof(T);
    dst_step /= sizeof(T);

    auto alpha = rvv<T>::vmv_v_x(typeid(T) == typeid(float) ? 1.0f : std::numeric_limits<T>::max(), rvv<T>::vsetvlmax());
    int vl;
    for (int i = start; i < end; i++)
    {
        for (int j = 0; j < width; j += vl)
        {
            vl = rvv<T>::vsetvl(width - j);
            auto vec_src = rvv<T>::vle(src + i * src_step + j, vl);
            rvv<T>::vsseg(dst + i * dst_step + j * dcn, dcn, vec_src, vec_src, vec_src, alpha, vl);
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int cvtGraytoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int depth, int dcn)
{
    switch (depth)
    {
    case CV_8U:
        return ThreadPool::parallel_for(height, -1, cvtGraytoBGR<uchar>, reinterpret_cast<const uchar*>(src_data), src_step, reinterpret_cast<uchar*>(dst_data), dst_step, width, dcn);
    case CV_16U:
        return ThreadPool::parallel_for(height, -1, cvtGraytoBGR<ushort>, reinterpret_cast<const ushort*>(src_data), src_step, reinterpret_cast<ushort*>(dst_data), dst_step, width, dcn);
    case CV_32F:
        return ThreadPool::parallel_for(height, -1, cvtGraytoBGR<float>, reinterpret_cast<const float*>(src_data), src_step, reinterpret_cast<float*>(dst_data), dst_step, width, dcn);
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}
} // cv::cv_hal_rvv::GraytoBGR

namespace BGRtoGray {
#undef cv_hal_cvtBGRtoGray
#define cv_hal_cvtBGRtoGray cv::cv_hal_rvv::BGRtoGray::cvtBGRtoGray

template<typename T> struct rvv;
template<> struct rvv<uchar>
{
    using T = vuint8m1_t;
    static constexpr uint B2Y = 3735, G2Y = 19235, R2Y = 9798;
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e8m1(a); }
    static inline void vlseg(const uchar* a, int b, T& c, T& d, T& e, size_t f)
    {
        if (b == 3)
        {
            auto x = __riscv_vlseg3e8_v_u8m1x3(a, f);
            c = __riscv_vget_v_u8m1x3_u8m1(x, 0), d = __riscv_vget_v_u8m1x3_u8m1(x, 1), e = __riscv_vget_v_u8m1x3_u8m1(x, 2);
        }
        else
        {
            auto x = __riscv_vlseg4e8_v_u8m1x4(a, f);
            c = __riscv_vget_v_u8m1x4_u8m1(x, 0), d = __riscv_vget_v_u8m1x4_u8m1(x, 1), e = __riscv_vget_v_u8m1x4_u8m1(x, 2);
        }
    }
    static inline void vse(uchar* a, T b, size_t c) { return __riscv_vse8(a, b, c); }
    static inline vuint32m4_t vcvt0(T a, size_t b) { return __riscv_vzext_vf4(a, b); }
    static inline T vcvt1(vuint32m4_t a, size_t b, size_t c) { return __riscv_vncvt_x(__riscv_vncvt_x(__riscv_vssrl(a, b, __RISCV_VXRM_RNU, c), c), c); }
    static inline vuint32m4_t vmul(vuint32m4_t a, uint b, size_t c) { return __riscv_vmul(a, b, c); }
    static inline vuint32m4_t vmadd(vuint32m4_t a, uint b, vuint32m4_t c, size_t d) { return __riscv_vmadd(a, b, c, d); }
};
template<> struct rvv<ushort>
{
    using T = vuint16m2_t;
    static constexpr uint B2Y = 3735, G2Y = 19235, R2Y = 9798;
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e16m2(a); }
    static inline void vlseg(const ushort* a, int b, T& c, T& d, T& e, size_t f)
    {
        if (b == 3)
        {
            auto x = __riscv_vlseg3e16_v_u16m2x3(a, f);
            c = __riscv_vget_v_u16m2x3_u16m2(x, 0), d = __riscv_vget_v_u16m2x3_u16m2(x, 1), e = __riscv_vget_v_u16m2x3_u16m2(x, 2);
        }
        else
        {
            auto x = __riscv_vlseg4e16_v_u16m2x4(a, f);
            c = __riscv_vget_v_u16m2x4_u16m2(x, 0), d = __riscv_vget_v_u16m2x4_u16m2(x, 1), e = __riscv_vget_v_u16m2x4_u16m2(x, 2);
        }
    }
    static inline void vse(ushort* a, T b, size_t c) { return __riscv_vse16(a, b, c); }
    static inline vuint32m4_t vcvt0(T a, size_t b) { return __riscv_vzext_vf2(a, b); }
    static inline T vcvt1(vuint32m4_t a, size_t b, size_t c) { return __riscv_vncvt_x(__riscv_vssrl(a, b, __RISCV_VXRM_RNU, c), c); }
    static inline vuint32m4_t vmul(vuint32m4_t a, uint b, size_t c) { return __riscv_vmul(a, b, c); }
    static inline vuint32m4_t vmadd(vuint32m4_t a, uint b, vuint32m4_t c, size_t d) { return __riscv_vmadd(a, b, c, d); }
};
template<> struct rvv<float>
{
    using T = vfloat32m2_t;
    static constexpr float B2Y = 0.114f, G2Y = 0.587f, R2Y = 0.299f;
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e32m2(a); }
    static inline void vlseg(const float* a, int b, T& c, T& d, T& e, size_t f)
    {
        if (b == 3)
        {
            auto x = __riscv_vlseg3e32_v_f32m2x3(a, f);
            c = __riscv_vget_v_f32m2x3_f32m2(x, 0), d = __riscv_vget_v_f32m2x3_f32m2(x, 1), e = __riscv_vget_v_f32m2x3_f32m2(x, 2);
        }
        else
        {
            auto x = __riscv_vlseg4e32_v_f32m2x4(a, f);
            c = __riscv_vget_v_f32m2x4_f32m2(x, 0), d = __riscv_vget_v_f32m2x4_f32m2(x, 1), e = __riscv_vget_v_f32m2x4_f32m2(x, 2);
        }
    }
    static inline void vse(float* a, T b, size_t c) { return __riscv_vse32(a, b, c); }
    static inline T vcvt0(T a, size_t) { return a; }
    static inline T vcvt1(T a, size_t, size_t) { return a; }
    static inline T vmul(T a, float b, size_t c) { return __riscv_vfmul(a, b, c); }
    static inline T vmadd(T a, float b, T c, size_t d) { return __riscv_vfmadd(a, b, c, d); }
};

template<typename T>
inline int cvtBGRtoGray(int start, int end, const T * src, size_t src_step, T * dst, size_t dst_step, int width, int scn, bool swapBlue)
{
    if (scn != 3 && scn != 4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    src_step /= sizeof(T);
    dst_step /= sizeof(T);

    int vl;
    for (int i = start; i < end; i++)
    {
        for (int j = 0; j < width; j += vl)
        {
            vl = rvv<T>::vsetvl(width - j);
            typename rvv<T>::T vec_srcB, vec_srcG, vec_srcR;
            rvv<T>::vlseg(src + i * src_step + j * scn, scn, vec_srcB, vec_srcG, vec_srcR, vl);
            if (swapBlue)
            {
                auto t = vec_srcB;
                vec_srcB = vec_srcR, vec_srcR = t;
            }
            auto vec_dst = rvv<T>::vmadd(rvv<T>::vcvt0(vec_srcB, vl), rvv<T>::B2Y, rvv<T>::vmadd(rvv<T>::vcvt0(vec_srcG, vl), rvv<T>::G2Y, rvv<T>::vmul(rvv<T>::vcvt0(vec_srcR, vl), rvv<T>::R2Y, vl), vl), vl);
            rvv<T>::vse(dst + i * dst_step + j, rvv<T>::vcvt1(vec_dst, 15, vl), vl);
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int cvtBGRtoGray(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int depth, int scn, bool swapBlue)
{
    switch (depth)
    {
    case CV_8U:
        return ThreadPool::parallel_for(height, -1, cvtBGRtoGray<uchar>, reinterpret_cast<const uchar*>(src_data), src_step, reinterpret_cast<uchar*>(dst_data), dst_step, width, scn, swapBlue);
    case CV_16U:
        return ThreadPool::parallel_for(height, -1, cvtBGRtoGray<ushort>, reinterpret_cast<const ushort*>(src_data), src_step, reinterpret_cast<ushort*>(dst_data), dst_step, width, scn, swapBlue);
    case CV_32F:
        return ThreadPool::parallel_for(height, -1, cvtBGRtoGray<float>, reinterpret_cast<const float*>(src_data), src_step, reinterpret_cast<float*>(dst_data), dst_step, width, scn, swapBlue);
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}
} // cv::cv_hal_rvv::BGRtoGray

namespace YUVtoBGR {
#undef cv_hal_cvtYUVtoBGR
#define cv_hal_cvtYUVtoBGR cv::cv_hal_rvv::YUVtoBGR::cvtYUVtoBGR

template<typename T> struct rvv;
template<> struct rvv<uchar>
{
    using T = vuint8m1_t;
    static constexpr int U2B = 33292, U2G = -6472, V2G = -9519, V2R = 18678, CB2B = 29049, CB2G = -5636, CR2G = -11698, CR2R = 22987;
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e8m1(); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e8m1(a); }
    static inline void vlseg(const uchar* a, T& b, T& c, T& d, size_t e){ auto x = __riscv_vlseg3e8_v_u8m1x3(a, e); b = __riscv_vget_v_u8m1x3_u8m1(x, 0), c = __riscv_vget_v_u8m1x3_u8m1(x, 1), d = __riscv_vget_v_u8m1x3_u8m1(x, 2); }
    static inline void vsseg(uchar* a, int b, T c, T d, T e, T f, size_t g) { return b == 3 ? __riscv_vsseg3e8(a, __riscv_vcreate_v_u8m1x3(c, d, e), g) : __riscv_vsseg4e8(a, __riscv_vcreate_v_u8m1x4(c, d, e, f), g); }
    static inline vint32m4_t vcvt0(T a, size_t b) { return __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4(a, b)); }
    static inline T vcvt1(vint32m4_t a, vint32m4_t b, size_t c, size_t d) { return __riscv_vnclipu(__riscv_vnclipu(__riscv_vreinterpret_v_i32m4_u32m4(__riscv_vmax(__riscv_vadd(__riscv_vssra(a, c, __RISCV_VXRM_RNU, d), b, d), 0, d)), 0, __RISCV_VXRM_RNU, d), 0, __RISCV_VXRM_RNU, d); }
    static inline vint32m4_t vsub(vint32m4_t a, int b, size_t c) { return __riscv_vsub(a, b, c); }
    static inline vint32m4_t vmul(vint32m4_t a, int b, size_t c) { return __riscv_vmul(a, b, c); }
    static inline vint32m4_t vmadd(vint32m4_t a, int b, vint32m4_t c, size_t d) { return __riscv_vmadd(a, b, c, d); }
    static inline T vmv_v_x(uchar a, size_t b) { return __riscv_vmv_v_x_u8m1(a, b); }
};
template<> struct rvv<ushort>
{
    using T = vuint16m2_t;
    static constexpr int U2B = 33292, U2G = -6472, V2G = -9519, V2R = 18678, CB2B = 29049, CB2G = -5636, CR2G = -11698, CR2R = 22987;
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e16m2(); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e16m2(a); }
    static inline void vlseg(const ushort* a, T& b, T& c, T& d, size_t e){ auto x = __riscv_vlseg3e16_v_u16m2x3(a, e); b = __riscv_vget_v_u16m2x3_u16m2(x, 0), c = __riscv_vget_v_u16m2x3_u16m2(x, 1), d = __riscv_vget_v_u16m2x3_u16m2(x, 2); }
    static inline void vsseg(ushort* a, int b, T c, T d, T e, T f, size_t g) { return b == 3 ? __riscv_vsseg3e16(a, __riscv_vcreate_v_u16m2x3(c, d, e), g) : __riscv_vsseg4e16(a, __riscv_vcreate_v_u16m2x4(c, d, e, f), g); }
    static inline vint32m4_t vcvt0(T a, size_t b) { return __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf2(a, b)); }
    static inline T vcvt1(vint32m4_t a, vint32m4_t b, size_t c, size_t d) { return __riscv_vnclipu(__riscv_vreinterpret_v_i32m4_u32m4(__riscv_vmax(__riscv_vadd(__riscv_vssra(a, c, __RISCV_VXRM_RNU, d), b, d), 0, d)), 0, __RISCV_VXRM_RNU, d); }
    static inline vint32m4_t vsub(vint32m4_t a, int b, size_t c) { return __riscv_vsub(a, b, c); }
    static inline vint32m4_t vmul(vint32m4_t a, int b, size_t c) { return __riscv_vmul(a, b, c); }
    static inline vint32m4_t vmadd(vint32m4_t a, int b, vint32m4_t c, size_t d) { return __riscv_vmadd(a, b, c, d); }
    static inline T vmv_v_x(ushort a, size_t b) { return __riscv_vmv_v_x_u16m2(a, b); }
};
template<> struct rvv<float>
{
    using T = vfloat32m2_t;
    static constexpr float U2B = 2.032f, U2G = -0.395f, V2G = -0.581f, V2R = 1.140f, CB2B = 1.773f, CB2G = -0.344f, CR2G = -0.714f, CR2R = 1.403f;
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e32m2(); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e32m2(a); }
    static inline void vlseg(const float* a, T& b, T& c, T& d, size_t e){ auto x = __riscv_vlseg3e32_v_f32m2x3(a, e); b = __riscv_vget_v_f32m2x3_f32m2(x, 0), c = __riscv_vget_v_f32m2x3_f32m2(x, 1), d = __riscv_vget_v_f32m2x3_f32m2(x, 2); }
    static inline void vsseg(float* a, int b, T c, T d, T e, T f, size_t g) { return b == 3 ? __riscv_vsseg3e32(a, __riscv_vcreate_v_f32m2x3(c, d, e), g) : __riscv_vsseg4e32(a, __riscv_vcreate_v_f32m2x4(c, d, e, f), g); }
    static inline T vcvt0(T a, size_t) { return a; }
    static inline T vcvt1(T a, T b, size_t, size_t d) { return __riscv_vfadd(a, b, d); }
    static inline T vsub(T a, float b, size_t c) { return __riscv_vfsub(a, b, c); }
    static inline T vmul(T a, float b, size_t c) { return __riscv_vfmul(a, b, c); }
    static inline T vmadd(T a, float b, T c, size_t d) { return __riscv_vfmadd(a, b, c, d); }
    static inline T vmv_v_x(float a, size_t b) { return __riscv_vfmv_v_f_f32m2(a, b); }
};

template<typename T>
inline int cvtYUVtoBGR(int start, int end, const T * src, size_t src_step, T * dst, size_t dst_step, int width, int dcn, bool swapBlue, bool isCbCr)
{
    if (dcn != 3 && dcn != 4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    src_step /= sizeof(T);
    dst_step /= sizeof(T);

    decltype(rvv<T>::U2B) delta = typeid(T) == typeid(float) ? 0.5f : std::numeric_limits<T>::max() / 2 + 1;
    auto alpha = rvv<T>::vmv_v_x(typeid(T) == typeid(float) ? 1.0f : std::numeric_limits<T>::max(), rvv<T>::vsetvlmax());
    int vl;
    for (int i = start; i < end; i++)
    {
        for (int j = 0; j < width; j += vl)
        {
            vl = rvv<T>::vsetvl(width - j);
            typename rvv<T>::T vec_srcY_T, vec_srcU_T, vec_srcV_T;
            rvv<T>::vlseg(src + i * src_step + j * 3, vec_srcY_T, vec_srcU_T, vec_srcV_T, vl);
            auto vec_srcY = rvv<T>::vcvt0(vec_srcY_T, vl);
            auto vec_srcU = rvv<T>::vcvt0(vec_srcU_T, vl);
            auto vec_srcV = rvv<T>::vcvt0(vec_srcV_T, vl);

            auto vec_dstB = rvv<T>::vmul(rvv<T>::vsub(vec_srcU, delta, vl), isCbCr ? rvv<T>::CB2B : rvv<T>::U2B, vl);
            auto vec_dstG = rvv<T>::vmul(rvv<T>::vsub(vec_srcU, delta, vl), isCbCr ? rvv<T>::CB2G : rvv<T>::U2G, vl);
            vec_dstG = rvv<T>::vmadd(rvv<T>::vsub(vec_srcV, delta, vl), isCbCr ? rvv<T>::CR2G : rvv<T>::V2G, vec_dstG, vl);
            auto vec_dstR = rvv<T>::vmul(rvv<T>::vsub(vec_srcV, delta, vl), isCbCr ? rvv<T>::CR2R : rvv<T>::V2R, vl);
            if (swapBlue)
            {
                auto t = vec_dstB;
                vec_dstB = vec_dstR, vec_dstR = t;
            }
            rvv<T>::vsseg(dst + i * dst_step + j * dcn, dcn, rvv<T>::vcvt1(vec_dstB, vec_srcY, 14, vl), rvv<T>::vcvt1(vec_dstG, vec_srcY, 14, vl), rvv<T>::vcvt1(vec_dstR, vec_srcY, 14, vl), alpha, vl);
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int cvtYUVtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int depth, int dcn, bool swapBlue, bool isCbCr)
{
    switch (depth)
    {
    case CV_8U:
        return ThreadPool::parallel_for(height, -1, cvtYUVtoBGR<uchar>, reinterpret_cast<const uchar*>(src_data), src_step, reinterpret_cast<uchar*>(dst_data), dst_step, width, dcn, swapBlue, isCbCr);
    case CV_16U:
        return ThreadPool::parallel_for(height, -1, cvtYUVtoBGR<ushort>, reinterpret_cast<const ushort*>(src_data), src_step, reinterpret_cast<ushort*>(dst_data), dst_step, width, dcn, swapBlue, isCbCr);
    case CV_32F:
        return ThreadPool::parallel_for(height, -1, cvtYUVtoBGR<float>, reinterpret_cast<const float*>(src_data), src_step, reinterpret_cast<float*>(dst_data), dst_step, width, dcn, swapBlue, isCbCr);
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}
} // cv::cv_hal_rvv::YUVtoBGR

namespace BGRtoYUV {
#undef cv_hal_cvtBGRtoYUV
#define cv_hal_cvtBGRtoYUV cv::cv_hal_rvv::BGRtoYUV::cvtBGRtoYUV

template<typename T> struct rvv;
template<> struct rvv<uchar>
{
    using T = vuint8m1_t;
    static constexpr int B2Y = 1868, G2Y = 9617, R2Y = 4899, B2U = 8061, R2V = 14369, YCB = 9241, YCR = 11682;
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e8m1(); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e8m1(a); }
    static inline void vlseg(const uchar* a, int b, T& c, T& d, T& e, size_t f)
    {
        if (b == 3)
        {
            auto x = __riscv_vlseg3e8_v_u8m1x3(a, f);
            c = __riscv_vget_v_u8m1x3_u8m1(x, 0), d = __riscv_vget_v_u8m1x3_u8m1(x, 1), e = __riscv_vget_v_u8m1x3_u8m1(x, 2);
        }
        else
        {
            auto x = __riscv_vlseg4e8_v_u8m1x4(a, f);
            c = __riscv_vget_v_u8m1x4_u8m1(x, 0), d = __riscv_vget_v_u8m1x4_u8m1(x, 1), e = __riscv_vget_v_u8m1x4_u8m1(x, 2);
        }
    }
    static inline void vsseg(uchar* a, T b, T c, T d, size_t e) { __riscv_vsseg3e8(a, __riscv_vcreate_v_u8m1x3(b, c, d), e); }
    static inline vint32m4_t vcvt0(T a, size_t b) { return __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4(a, b)); }
    static inline T vcvt1(vint32m4_t a, size_t b, size_t c) { return __riscv_vnclipu(__riscv_vnclipu(__riscv_vreinterpret_v_i32m4_u32m4(__riscv_vmax(a, 0, c)), b, __RISCV_VXRM_RNU, c), 0, __RISCV_VXRM_RNU, c); }
    static inline vint32m4_t vssra(vint32m4_t a, size_t b, size_t c) { return __riscv_vssra(a, b, __RISCV_VXRM_RNU, c); }
    static inline vint32m4_t vsub(vint32m4_t a, vint32m4_t b, size_t c) { return __riscv_vsub(a, b, c); }
    static inline vint32m4_t vmul(vint32m4_t a, int b, size_t c) { return __riscv_vmul(a, b, c); }
    static inline vint32m4_t vmadd(vint32m4_t a, int b, vint32m4_t c, size_t d) { return __riscv_vmadd(a, b, c, d); }
    static inline vint32m4_t vmv_v_x(int a, size_t b) { return __riscv_vmv_v_x_i32m4(a, b); }
};
template<> struct rvv<ushort>
{
    using T = vuint16m2_t;
    static constexpr int B2Y = 1868, G2Y = 9617, R2Y = 4899, B2U = 8061, R2V = 14369, YCB = 9241, YCR = 11682;
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e16m2(); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e16m2(a); }
    static inline void vlseg(const ushort* a, int b, T& c, T& d, T& e, size_t f)
    {
        if (b == 3)
        {
            auto x = __riscv_vlseg3e16_v_u16m2x3(a, f);
            c = __riscv_vget_v_u16m2x3_u16m2(x, 0), d = __riscv_vget_v_u16m2x3_u16m2(x, 1), e = __riscv_vget_v_u16m2x3_u16m2(x, 2);
        }
        else
        {
            auto x = __riscv_vlseg4e16_v_u16m2x4(a, f);
            c = __riscv_vget_v_u16m2x4_u16m2(x, 0), d = __riscv_vget_v_u16m2x4_u16m2(x, 1), e = __riscv_vget_v_u16m2x4_u16m2(x, 2);
        }
    }
    static inline void vsseg(ushort* a, T b, T c, T d, size_t e) { __riscv_vsseg3e16(a, __riscv_vcreate_v_u16m2x3(b, c, d), e); }
    static inline vint32m4_t vcvt0(T a, size_t b) { return __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf2(a, b)); }
    static inline T vcvt1(vint32m4_t a, size_t b, size_t c) { return __riscv_vnclipu(__riscv_vreinterpret_v_i32m4_u32m4(__riscv_vmax(a, 0, c)), b, __RISCV_VXRM_RNU, c); }
    static inline vint32m4_t vssra(vint32m4_t a, size_t b, size_t c) { return __riscv_vssra(a, b, __RISCV_VXRM_RNU, c); }
    static inline vint32m4_t vsub(vint32m4_t a, vint32m4_t b, size_t c) { return __riscv_vsub(a, b, c); }
    static inline vint32m4_t vmul(vint32m4_t a, int b, size_t c) { return __riscv_vmul(a, b, c); }
    static inline vint32m4_t vmadd(vint32m4_t a, int b, vint32m4_t c, size_t d) { return __riscv_vmadd(a, b, c, d); }
    static inline vint32m4_t vmv_v_x(int a, size_t b) { return __riscv_vmv_v_x_i32m4(a, b); }
};
template<> struct rvv<float>
{
    using T = vfloat32m2_t;
    static constexpr float B2Y = 0.114f, G2Y = 0.587f, R2Y = 0.299f, B2U = 0.492f, R2V = 0.877f, YCB = 0.564f, YCR = 0.713f;
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e32m2(); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e32m2(a); }
    static inline void vlseg(const float* a, int b, T& c, T& d, T& e, size_t f)
    {
        if (b == 3)
        {
            auto x = __riscv_vlseg3e32_v_f32m2x3(a, f);
            c = __riscv_vget_v_f32m2x3_f32m2(x, 0), d = __riscv_vget_v_f32m2x3_f32m2(x, 1), e = __riscv_vget_v_f32m2x3_f32m2(x, 2);
        }
        else
        {
            auto x = __riscv_vlseg4e32_v_f32m2x4(a, f);
            c = __riscv_vget_v_f32m2x4_f32m2(x, 0), d = __riscv_vget_v_f32m2x4_f32m2(x, 1), e = __riscv_vget_v_f32m2x4_f32m2(x, 2);
        }
    }
    static inline void vsseg(float* a, T b, T c, T d, size_t e) { __riscv_vsseg3e32(a, __riscv_vcreate_v_f32m2x3(b, c, d), e); }
    static inline T vcvt0(T a, size_t) { return a; }
    static inline T vcvt1(T a, size_t, size_t) { return a; }
    static inline T vssra(T a, size_t, size_t) { return a; }
    static inline T vsub(T a, T b, size_t c) { return __riscv_vfsub(a, b, c); }
    static inline T vmul(T a, float b, size_t c) { return __riscv_vfmul(a, b, c); }
    static inline T vmadd(T a, float b, T c, size_t d) { return __riscv_vfmadd(a, b, c, d); }
    static inline T vmv_v_x(float a, size_t b) { return __riscv_vfmv_v_f_f32m2(a, b); }
};

template<typename T>
inline int cvtBGRtoYUV(int start, int end, const T * src, size_t src_step, T * dst, size_t dst_step, int width, int scn, bool swapBlue, bool isCbCr)
{
    if (scn != 3 && scn != 4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    src_step /= sizeof(T);
    dst_step /= sizeof(T);

    auto delta = rvv<T>::vmv_v_x((1 << 14) * (typeid(T) == typeid(float) ? 0.5f : std::numeric_limits<T>::max() / 2 + 1), rvv<T>::vsetvlmax());
    int vl;
    for (int i = start; i < end; i++)
    {
        for (int j = 0; j < width; j += vl)
        {
            vl = rvv<T>::vsetvl(width - j);
            typename rvv<T>::T vec_srcB_T, vec_srcG_T, vec_srcR_T;
            rvv<T>::vlseg(src + i * src_step + j * scn, scn, vec_srcB_T, vec_srcG_T, vec_srcR_T, vl);
            auto vec_srcB = rvv<T>::vcvt0(vec_srcB_T, vl);
            auto vec_srcG = rvv<T>::vcvt0(vec_srcG_T, vl);
            auto vec_srcR = rvv<T>::vcvt0(vec_srcR_T, vl);
            if (swapBlue)
            {
                auto t = vec_srcB;
                vec_srcB = vec_srcR, vec_srcR = t;
            }

            auto vec_dstY = rvv<T>::vssra(rvv<T>::vmadd(vec_srcB, rvv<T>::B2Y, rvv<T>::vmadd(vec_srcG, rvv<T>::G2Y, rvv<T>::vmul(vec_srcR, rvv<T>::R2Y, vl), vl), vl), 14, vl);
            auto vec_dstU = rvv<T>::vmadd(rvv<T>::vsub(vec_srcB, vec_dstY, vl), isCbCr ? rvv<T>::YCB : rvv<T>::B2U, delta, vl);
            auto vec_dstV = rvv<T>::vmadd(rvv<T>::vsub(vec_srcR, vec_dstY, vl), isCbCr ? rvv<T>::YCR : rvv<T>::R2V, delta, vl);
            rvv<T>::vsseg(dst + i * dst_step + j * 3, rvv<T>::vcvt1(vec_dstY, 0, vl), rvv<T>::vcvt1(vec_dstU, 14, vl), rvv<T>::vcvt1(vec_dstV, 14, vl), vl);
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int cvtBGRtoYUV(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int depth, int scn, bool swapBlue, bool isCbCr)
{
    switch (depth)
    {
    case CV_8U:
        return ThreadPool::parallel_for(height, -1, cvtBGRtoYUV<uchar>, reinterpret_cast<const uchar*>(src_data), src_step, reinterpret_cast<uchar*>(dst_data), dst_step, width, scn, swapBlue, isCbCr);
    case CV_16U:
        return ThreadPool::parallel_for(height, -1, cvtBGRtoYUV<ushort>, reinterpret_cast<const ushort*>(src_data), src_step, reinterpret_cast<ushort*>(dst_data), dst_step, width, scn, swapBlue, isCbCr);
    case CV_32F:
        return ThreadPool::parallel_for(height, -1, cvtBGRtoYUV<float>, reinterpret_cast<const float*>(src_data), src_step, reinterpret_cast<float*>(dst_data), dst_step, width, scn, swapBlue, isCbCr);
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}
} // cv::cv_hal_rvv::BGRtoYUV

}}

#endif
