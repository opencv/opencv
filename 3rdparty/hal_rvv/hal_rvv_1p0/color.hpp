// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_COLOR_HPP_INCLUDED
#define OPENCV_HAL_RVV_COLOR_HPP_INCLUDED

#include <riscv_vector.h>
#include "thread_pool.hpp"

namespace cv { namespace cv_hal_rvv {

namespace color {
    template<typename... Args>
    static inline int invoke(int length, double fstripe, std::function<int(int, int, Args...)> func, Args&&... args)
    {
        if (length < 240)
            return func(0, length, std::forward<Args>(args)...);
        return ThreadPool::parallel_for(length, fstripe, func, std::forward<Args>(args)...);
    }
} // cv::cv_hal_rvv::color

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
static inline int cvtBGRtoBGR(int start, int end, const T * src, size_t src_step, T * dst, size_t dst_step, int width, int scn, int dcn, bool swapBlue)
{
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
        for (int i = start; i < end; i++)
        {
            int vl;
            for (int j = 0; j < width; j += vl)
            {
                vl = rvv<T>::vsetvl(width - j);
                typename rvv<T>::T vec_srcB, vec_srcG, vec_srcR, vec_srcA{};
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
    if ((scn != 3 && scn != 4) || (dcn != 3 && dcn != 4))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    switch (depth)
    {
    case CV_8U:
        return color::invoke(height, -1, cvtBGRtoBGR<uchar>, reinterpret_cast<const uchar*>(src_data), src_step, reinterpret_cast<uchar*>(dst_data), dst_step, width, scn, dcn, swapBlue);
    case CV_16U:
        return color::invoke(height, -1, cvtBGRtoBGR<ushort>, reinterpret_cast<const ushort*>(src_data), src_step, reinterpret_cast<ushort*>(dst_data), dst_step, width, scn, dcn, swapBlue);
    case CV_32F:
        return color::invoke(height, -1, cvtBGRtoBGR<float>, reinterpret_cast<const float*>(src_data), src_step, reinterpret_cast<float*>(dst_data), dst_step, width, scn, dcn, swapBlue);
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
static inline int cvtGraytoBGR(int start, int end, const T * src, size_t src_step, T * dst, size_t dst_step, int width, int dcn)
{
    src_step /= sizeof(T);
    dst_step /= sizeof(T);

    auto alpha = rvv<T>::vmv_v_x(typeid(T) == typeid(float) ? 1.0f : std::numeric_limits<T>::max(), rvv<T>::vsetvlmax());
    for (int i = start; i < end; i++)
    {
        int vl;
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
    if (dcn != 3 && dcn != 4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    switch (depth)
    {
    case CV_8U:
        return color::invoke(height, -1, cvtGraytoBGR<uchar>, reinterpret_cast<const uchar*>(src_data), src_step, reinterpret_cast<uchar*>(dst_data), dst_step, width, dcn);
    case CV_16U:
        return color::invoke(height, -1, cvtGraytoBGR<ushort>, reinterpret_cast<const ushort*>(src_data), src_step, reinterpret_cast<ushort*>(dst_data), dst_step, width, dcn);
    case CV_32F:
        return color::invoke(height, -1, cvtGraytoBGR<float>, reinterpret_cast<const float*>(src_data), src_step, reinterpret_cast<float*>(dst_data), dst_step, width, dcn);
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
static inline int cvtBGRtoGray(int start, int end, const T * src, size_t src_step, T * dst, size_t dst_step, int width, int scn, bool swapBlue)
{
    src_step /= sizeof(T);
    dst_step /= sizeof(T);

    for (int i = start; i < end; i++)
    {
        int vl;
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
    if (scn != 3 && scn != 4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    switch (depth)
    {
    case CV_8U:
        return color::invoke(height, -1, cvtBGRtoGray<uchar>, reinterpret_cast<const uchar*>(src_data), src_step, reinterpret_cast<uchar*>(dst_data), dst_step, width, scn, swapBlue);
    case CV_16U:
        return color::invoke(height, -1, cvtBGRtoGray<ushort>, reinterpret_cast<const ushort*>(src_data), src_step, reinterpret_cast<ushort*>(dst_data), dst_step, width, scn, swapBlue);
    case CV_32F:
        return color::invoke(height, -1, cvtBGRtoGray<float>, reinterpret_cast<const float*>(src_data), src_step, reinterpret_cast<float*>(dst_data), dst_step, width, scn, swapBlue);
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
static inline int cvtYUVtoBGR(int start, int end, const T * src, size_t src_step, T * dst, size_t dst_step, int width, int dcn, bool swapBlue, bool isCbCr)
{
    src_step /= sizeof(T);
    dst_step /= sizeof(T);

    decltype(rvv<T>::U2B) delta = typeid(T) == typeid(float) ? 0.5f : std::numeric_limits<T>::max() / 2 + 1;
    auto alpha = rvv<T>::vmv_v_x(typeid(T) == typeid(float) ? 1.0f : std::numeric_limits<T>::max(), rvv<T>::vsetvlmax());
    for (int i = start; i < end; i++)
    {
        int vl;
        for (int j = 0; j < width; j += vl)
        {
            vl = rvv<T>::vsetvl(width - j);
            typename rvv<T>::T vec_srcY_T, vec_srcU_T, vec_srcV_T;
            rvv<T>::vlseg(src + i * src_step + j * 3, vec_srcY_T, vec_srcU_T, vec_srcV_T, vl);
            auto vec_srcY = rvv<T>::vcvt0(vec_srcY_T, vl);
            auto vec_srcU = rvv<T>::vcvt0(vec_srcU_T, vl);
            auto vec_srcV = rvv<T>::vcvt0(vec_srcV_T, vl);
            if (isCbCr)
            {
                auto t = vec_srcU;
                vec_srcU = vec_srcV, vec_srcV = t;
            }

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
    if (dcn != 3 && dcn != 4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    switch (depth)
    {
    case CV_8U:
        return color::invoke(height, -1, cvtYUVtoBGR<uchar>, reinterpret_cast<const uchar*>(src_data), src_step, reinterpret_cast<uchar*>(dst_data), dst_step, width, dcn, swapBlue, isCbCr);
    case CV_16U:
        return color::invoke(height, -1, cvtYUVtoBGR<ushort>, reinterpret_cast<const ushort*>(src_data), src_step, reinterpret_cast<ushort*>(dst_data), dst_step, width, dcn, swapBlue, isCbCr);
    case CV_32F:
        return color::invoke(height, -1, cvtYUVtoBGR<float>, reinterpret_cast<const float*>(src_data), src_step, reinterpret_cast<float*>(dst_data), dst_step, width, dcn, swapBlue, isCbCr);
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
static inline int cvtBGRtoYUV(int start, int end, const T * src, size_t src_step, T * dst, size_t dst_step, int width, int scn, bool swapBlue, bool isCbCr)
{
    src_step /= sizeof(T);
    dst_step /= sizeof(T);

    auto delta = rvv<T>::vmv_v_x(typeid(T) == typeid(float) ? 0.5f : (1 << 14) * (std::numeric_limits<T>::max() / 2 + 1), rvv<T>::vsetvlmax());
    for (int i = start; i < end; i++)
    {
        int vl;
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
            if (isCbCr)
            {
                auto t = vec_dstU;
                vec_dstU = vec_dstV, vec_dstV = t;
            }
            rvv<T>::vsseg(dst + i * dst_step + j * 3, rvv<T>::vcvt1(vec_dstY, 0, vl), rvv<T>::vcvt1(vec_dstU, 14, vl), rvv<T>::vcvt1(vec_dstV, 14, vl), vl);
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int cvtBGRtoYUV(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int depth, int scn, bool swapBlue, bool isCbCr)
{
    if (scn != 3 && scn != 4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    switch (depth)
    {
    case CV_8U:
        return color::invoke(height, -1, cvtBGRtoYUV<uchar>, reinterpret_cast<const uchar*>(src_data), src_step, reinterpret_cast<uchar*>(dst_data), dst_step, width, scn, swapBlue, isCbCr);
    case CV_16U:
        return color::invoke(height, -1, cvtBGRtoYUV<ushort>, reinterpret_cast<const ushort*>(src_data), src_step, reinterpret_cast<ushort*>(dst_data), dst_step, width, scn, swapBlue, isCbCr);
    case CV_32F:
        return color::invoke(height, -1, cvtBGRtoYUV<float>, reinterpret_cast<const float*>(src_data), src_step, reinterpret_cast<float*>(dst_data), dst_step, width, scn, swapBlue, isCbCr);
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}
} // cv::cv_hal_rvv::BGRtoYUV

namespace PlaneYUVtoBGR {
#undef cv_hal_cvtOnePlaneYUVtoBGR
#define cv_hal_cvtOnePlaneYUVtoBGR cv::cv_hal_rvv::PlaneYUVtoBGR::cvtOnePlaneYUVtoBGR
#undef cv_hal_cvtTwoPlaneYUVtoBGR
#define cv_hal_cvtTwoPlaneYUVtoBGR cv::cv_hal_rvv::PlaneYUVtoBGR::cvtTwoPlaneYUVtoBGR
#undef cv_hal_cvtThreePlaneYUVtoBGR
#define cv_hal_cvtThreePlaneYUVtoBGR cv::cv_hal_rvv::PlaneYUVtoBGR::cvtThreePlaneYUVtoBGR

static const int ITUR_BT_601_SHIFT = 20;
static const int ITUR_BT_601_CY  = 1220542;
static const int ITUR_BT_601_CUB = 2116026;
static const int ITUR_BT_601_CUG = -409993;
static const int ITUR_BT_601_CVG = -852492;
static const int ITUR_BT_601_CVR = 1673527;

static inline void uvToBGRuv(int vl, const vuint8m1_t u, const vuint8m1_t v, vint32m4_t& buv, vint32m4_t& guv, vint32m4_t& ruv)
{
    auto uu = __riscv_vsub(__riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4(u, vl)), 128, vl);
    auto vv = __riscv_vsub(__riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4(v, vl)), 128, vl);

    auto shift = __riscv_vmv_v_x_i32m4(1 << (ITUR_BT_601_SHIFT - 1), vl);
    buv = __riscv_vmadd(uu, ITUR_BT_601_CUB, shift, vl);
    guv = __riscv_vmadd(uu, ITUR_BT_601_CUG, __riscv_vmadd(vv, ITUR_BT_601_CVG, shift, vl), vl);
    ruv = __riscv_vmadd(vv, ITUR_BT_601_CVR, shift, vl);
}

static inline void yBGRuvToBGRA(int vl, const vuint8m1_t vy, const vint32m4_t buv, const vint32m4_t guv, const vint32m4_t ruv,
                                vuint8m1_t& b, vuint8m1_t& g, vuint8m1_t& r, vuint8m1_t& a)
{
    auto yy = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4(vy, vl));
    auto y = __riscv_vmul(__riscv_vmax(__riscv_vsub(yy, 16, vl), 0, vl), ITUR_BT_601_CY, vl);
    b = __riscv_vnclipu(__riscv_vreinterpret_v_i16m2_u16m2(__riscv_vmax(__riscv_vnclip(__riscv_vadd(y, buv, vl), ITUR_BT_601_SHIFT, __RISCV_VXRM_RDN, vl), 0, vl)), 0, __RISCV_VXRM_RDN, vl);
    g = __riscv_vnclipu(__riscv_vreinterpret_v_i16m2_u16m2(__riscv_vmax(__riscv_vnclip(__riscv_vadd(y, guv, vl), ITUR_BT_601_SHIFT, __RISCV_VXRM_RDN, vl), 0, vl)), 0, __RISCV_VXRM_RDN, vl);
    r = __riscv_vnclipu(__riscv_vreinterpret_v_i16m2_u16m2(__riscv_vmax(__riscv_vnclip(__riscv_vadd(y, ruv, vl), ITUR_BT_601_SHIFT, __RISCV_VXRM_RDN, vl), 0, vl)), 0, __RISCV_VXRM_RDN, vl);
    a = __riscv_vmv_v_x_u8m1(0xFF, vl);
}

static inline void cvtYuv42xxp2BGR8(int vl, const vuint8m1_t u, const vuint8m1_t v,
                                    const vuint8m1_t vy01, const vuint8m1_t vy11, const vuint8m1_t vy02, const vuint8m1_t vy12,
                                    uchar* row1, uchar* row2, int dcn, bool swapBlue)
{
    vint32m4_t buv, guv, ruv;
    uvToBGRuv(vl, u, v, buv, guv, ruv);

    auto cvt = [&](const vuint8m1_t vy0, const vuint8m1_t vy1, uchar* row) {
        vuint8m1_t b0, g0, r0, a0;
        vuint8m1_t b1, g1, r1, a1;

        yBGRuvToBGRA(vl, vy0, buv, guv, ruv, b0, g0, r0, a0);
        yBGRuvToBGRA(vl, vy1, buv, guv, ruv, b1, g1, r1, a1);
        if (swapBlue)
        {
            auto t = b0;
            b0 = r0, r0 = t;
            t = b1, b1 = r1, r1 = t;
        }

        if (dcn == 3)
        {
            vuint8m1x6_t x{};
            x = __riscv_vset_v_u8m1_u8m1x6(x, 0, b0);
            x = __riscv_vset_v_u8m1_u8m1x6(x, 1, g0);
            x = __riscv_vset_v_u8m1_u8m1x6(x, 2, r0);
            x = __riscv_vset_v_u8m1_u8m1x6(x, 3, b1);
            x = __riscv_vset_v_u8m1_u8m1x6(x, 4, g1);
            x = __riscv_vset_v_u8m1_u8m1x6(x, 5, r1);
            __riscv_vsseg6e8(row, x, vl);
        }
        else
        {
            vuint8m1x8_t x{};
            x = __riscv_vset_v_u8m1_u8m1x8(x, 0, b0);
            x = __riscv_vset_v_u8m1_u8m1x8(x, 1, g0);
            x = __riscv_vset_v_u8m1_u8m1x8(x, 2, r0);
            x = __riscv_vset_v_u8m1_u8m1x8(x, 3, a0);
            x = __riscv_vset_v_u8m1_u8m1x8(x, 4, b1);
            x = __riscv_vset_v_u8m1_u8m1x8(x, 5, g1);
            x = __riscv_vset_v_u8m1_u8m1x8(x, 6, r1);
            x = __riscv_vset_v_u8m1_u8m1x8(x, 7, a1);
            __riscv_vsseg8e8(row, x, vl);
        }
    };

    cvt(vy01, vy11, row1);
    if (row2)
        cvt(vy02, vy12, row2);
}

// the algorithm is copied from imgproc/src/color_yuv.simd.cpp,
// in the functor struct YUV422toRGB8Invoker
static inline int cvtSinglePlaneYUVtoBGR(int start, int end, uchar * dst_data, size_t dst_step, int dst_width, size_t stride, const uchar* src_data, int dcn, bool swapBlue, int uIdx, int yIdx)
{
    // [yIdx, uIdx] | [uidx, vidx]:
    //     0, 0     |     1, 3
    //     0, 1     |     3, 1
    //     1, 0     |     0, 2
    const int uidx = 1 - yIdx + uIdx * 2;
    const int vidx = (2 + uidx) % 4;
    const uchar* yuv_src = src_data + start * stride;

    auto vget = [](vuint8m1x4_t x, size_t p) {
        switch (p)
        {
        case 0:
            return __riscv_vget_v_u8m1x4_u8m1(x, 0);
        case 1:
            return __riscv_vget_v_u8m1x4_u8m1(x, 1);
        case 2:
            return __riscv_vget_v_u8m1x4_u8m1(x, 2);
        case 3:
            return __riscv_vget_v_u8m1x4_u8m1(x, 3);
        }
        /*NOTREACHED*/
    };

    for (int j = start; j < end; j++, yuv_src += stride)
    {
        uchar* row = dst_data + dst_step * j;
        int vl;
        for (int i = 0; i < dst_width / 2; i += vl, row += vl*dcn*2)
        {
            vl = __riscv_vsetvl_e8m1(dst_width / 2 - i);
            auto x = __riscv_vlseg4e8_v_u8m1x4(yuv_src + 4 * i, vl);
            auto u = vget(x, uidx), v = vget(x, vidx), vy0 = vget(x, yIdx), vy1 = vget(x, yIdx + 2);

            cvtYuv42xxp2BGR8(vl, u, v, vy0, vy1, vuint8m1_t(), vuint8m1_t(), row, (uchar*)(0), dcn, swapBlue);
        }
    }

    return CV_HAL_ERROR_OK;
}

// the algorithm is copied from imgproc/src/color_yuv.simd.cpp,
// in the functor struct YUV420sp2RGB8Invoker and YUV420p2RGB8Invoker
static inline int cvtMultiPlaneYUVtoBGR(int start, int end, uchar * dst_data, size_t dst_step, int dst_width, size_t stride, const uchar* y1, const uchar* u, const uchar* v, int ustepIdx, int vstepIdx, int dcn, bool swapBlue, int uIdx)
{
    const int rangeBegin = start * 2;
    const int rangeEnd = end * 2;
    const uchar* my1 = y1 + rangeBegin * stride;

    int uvsteps[2] = {dst_width/2, static_cast<int>(stride) - dst_width/2};
    int usIdx = ustepIdx, vsIdx = vstepIdx;

    const uchar* u1 = u + (start / 2) * stride;
    const uchar* v1 = v + (start / 2) * stride;

    if (start % 2 == 1)
    {
        u1 += uvsteps[(usIdx++) & 1];
        v1 += uvsteps[(vsIdx++) & 1];
    }

    if (uIdx != -1)
    {
        // Overwrite u1 as uv in TwoPlane mode
        u1 = u + rangeBegin * stride / 2;
        uvsteps[0] = uvsteps[1] = stride;
    }

    for (int j = rangeBegin; j < rangeEnd; j += 2, my1 += stride * 2, u1 += uvsteps[(usIdx++) & 1], v1 += uvsteps[(vsIdx++) & 1])
    {
        uchar* row1 = dst_data + dst_step * j;
        uchar* row2 = dst_data + dst_step * (j + 1);
        const uchar* my2 = my1 + stride;

        int vl;
        for (int i = 0; i < dst_width / 2; i += vl, row1 += vl*dcn*2, row2 += vl*dcn*2)
        {
            vl = __riscv_vsetvl_e8m1(dst_width / 2 - i);
            auto x = __riscv_vlseg2e8_v_u8m1x2(my1 + 2 * i, vl);
            auto vy01 = __riscv_vget_v_u8m1x2_u8m1(x, 0), vy11 = __riscv_vget_v_u8m1x2_u8m1(x, 1);
            x = __riscv_vlseg2e8_v_u8m1x2(my2 + 2 * i, vl);
            auto vy02 = __riscv_vget_v_u8m1x2_u8m1(x, 0), vy12 = __riscv_vget_v_u8m1x2_u8m1(x, 1);

            vuint8m1_t uu, vv;
            switch (uIdx)
            {
            case 0:
                x = __riscv_vlseg2e8_v_u8m1x2(u1 + 2 * i, vl);
                uu = __riscv_vget_v_u8m1x2_u8m1(x, 0), vv = __riscv_vget_v_u8m1x2_u8m1(x, 1);
                break;
            case 1:
                x = __riscv_vlseg2e8_v_u8m1x2(u1 + 2 * i, vl);
                uu = __riscv_vget_v_u8m1x2_u8m1(x, 1), vv = __riscv_vget_v_u8m1x2_u8m1(x, 0);
                break;
            default:
                uu = __riscv_vle8_v_u8m1(u1 + i, vl), vv = __riscv_vle8_v_u8m1(v1 + i, vl);
            }

            cvtYuv42xxp2BGR8(vl, uu, vv, vy01, vy11, vy02, vy12, row1, row2, dcn, swapBlue);
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int cvtOnePlaneYUVtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int dst_width, int dst_height, int dcn, bool swapBlue, int uIdx, int yIdx)
{
    if (dcn != 3 && dcn != 4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    return color::invoke(dst_height, -1, {cvtSinglePlaneYUVtoBGR}, dst_data, dst_step, dst_width, src_step, src_data, dcn, swapBlue, uIdx, yIdx);
}

inline int cvtTwoPlaneYUVtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int dst_width, int dst_height, int dcn, bool swapBlue, int uIdx)
{
    if (dcn != 3 && dcn != 4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    const uchar* uv = src_data + src_step * static_cast<size_t>(dst_height);
    return color::invoke(dst_height / 2, -1, {cvtMultiPlaneYUVtoBGR}, dst_data, dst_step, dst_width, src_step, src_data, uv, uv, 0, 0, dcn, swapBlue, uIdx);
}

inline int cvtThreePlaneYUVtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int dst_width, int dst_height, int dcn, bool swapBlue, int uIdx)
{
    if (dcn != 3 && dcn != 4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    const uchar* u = src_data + src_step * static_cast<size_t>(dst_height);
    const uchar* v = src_data + src_step * static_cast<size_t>(dst_height + dst_height/4) + (dst_width/2) * ((dst_height % 4)/2);

    int ustepIdx = 0;
    int vstepIdx = dst_height % 4 == 2 ? 1 : 0;
    if(uIdx == 1) { std::swap(u ,v), std::swap(ustepIdx, vstepIdx); }

    return color::invoke(dst_height / 2, -1, {cvtMultiPlaneYUVtoBGR}, dst_data, dst_step, dst_width, src_step, src_data, u, v, ustepIdx, vstepIdx, dcn, swapBlue, -1);
}
} // cv::cv_hal_rvv::PlaneYUVtoBGR

namespace PlaneBGRtoYUV {
#undef cv_hal_cvtOnePlaneBGRtoYUV
#define cv_hal_cvtOnePlaneBGRtoYUV cv::cv_hal_rvv::PlaneBGRtoYUV::cvtOnePlaneBGRtoYUV
#undef cv_hal_cvtBGRtoTwoPlaneYUV
#define cv_hal_cvtBGRtoTwoPlaneYUV cv::cv_hal_rvv::PlaneBGRtoYUV::cvtBGRtoTwoPlaneYUV
#undef cv_hal_cvtBGRtoThreePlaneYUV
#define cv_hal_cvtBGRtoThreePlaneYUV cv::cv_hal_rvv::PlaneBGRtoYUV::cvtBGRtoThreePlaneYUV

static const int ITUR_BT_601_SHIFT = 20;
static const int ITUR_BT_601_CBY =  102760; // 0.114035 * (236-16)/256 * (1 << ITUR_BT_601_SHIFT)
static const int ITUR_BT_601_CGY =  528482; // 0.586472 * (236-16)/256 * (1 << ITUR_BT_601_SHIFT)
static const int ITUR_BT_601_CRY =  269484; // 0.299055 * (236-16)/256 * (1 << ITUR_BT_601_SHIFT)
static const int ITUR_BT_601_CBU =  460324; //  0.439 * (1 << (ITUR_BT_601_SHIFT-1))
static const int ITUR_BT_601_CGU = -305135; // -0.291 * (1 << (ITUR_BT_601_SHIFT-1))
static const int ITUR_BT_601_CRU = -155188; // -0.148 * (1 << (ITUR_BT_601_SHIFT-1))
static const int ITUR_BT_601_CBV =  -74448; // -0.071 * (1 << (ITUR_BT_601_SHIFT-1))
static const int ITUR_BT_601_CGV = -385875; // -0.368 * (1 << (ITUR_BT_601_SHIFT-1))

static inline vuint8m1_t bgrToY42x(int vl, vuint8m1_t b, vuint8m1_t g, vuint8m1_t r)
{
    auto bb = __riscv_vzext_vf4(b, vl);
    auto gg = __riscv_vzext_vf4(g, vl);
    auto rr = __riscv_vzext_vf4(r, vl);
    auto yy = __riscv_vmadd(bb, ITUR_BT_601_CBY, __riscv_vmadd(gg, ITUR_BT_601_CGY, __riscv_vmadd(rr, ITUR_BT_601_CRY, __riscv_vmv_v_x_u32m4((16 << ITUR_BT_601_SHIFT) + (1 << (ITUR_BT_601_SHIFT - 1)), vl), vl), vl), vl);
    return __riscv_vnclipu(__riscv_vnclipu(yy, ITUR_BT_601_SHIFT, __RISCV_VXRM_RDN, vl), 0, __RISCV_VXRM_RDN, vl);
}

static inline void bgrToUV42x(int vl, vuint8m1_t b, vuint8m1_t g, vuint8m1_t r, vuint8m1_t& u, vuint8m1_t& v)
{
    auto bb = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4(b, vl));
    auto gg = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4(g, vl));
    auto rr = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4(r, vl));
    auto uu = __riscv_vmadd(bb, ITUR_BT_601_CBU, __riscv_vmadd(gg, ITUR_BT_601_CGU, __riscv_vmadd(rr, ITUR_BT_601_CRU, __riscv_vmv_v_x_i32m4((128 << ITUR_BT_601_SHIFT) + (1 << (ITUR_BT_601_SHIFT - 1)), vl), vl), vl), vl);
    auto vv = __riscv_vmadd(bb, ITUR_BT_601_CBV, __riscv_vmadd(gg, ITUR_BT_601_CGV, __riscv_vmadd(rr, ITUR_BT_601_CBU, __riscv_vmv_v_x_i32m4((128 << ITUR_BT_601_SHIFT) + (1 << (ITUR_BT_601_SHIFT - 1)), vl), vl), vl), vl);
    u = __riscv_vnclipu(__riscv_vreinterpret_v_i16m2_u16m2(__riscv_vmax(__riscv_vnclip(uu, ITUR_BT_601_SHIFT, __RISCV_VXRM_RDN, vl), 0, vl)), 0, __RISCV_VXRM_RDN, vl);
    v = __riscv_vnclipu(__riscv_vreinterpret_v_i16m2_u16m2(__riscv_vmax(__riscv_vnclip(vv, ITUR_BT_601_SHIFT, __RISCV_VXRM_RDN, vl), 0, vl)), 0, __RISCV_VXRM_RDN, vl);
}

static const int BGR2YUV422_SHIFT = 14;
static const int B2Y422 =  1606; // 0.114062 * (236 - 16) / 256 * 16384
static const int G2Y422 =  8258; // 0.586506 * (236 - 16) / 256 * 16384
static const int R2Y422 =  4211; // 0.299077 * (236 - 16) / 256 * 16384
static const int B2U422 =  3596; //  0.439 * 8192
static const int G2U422 = -2384; // -0.291 * 8192
static const int R2U422 = -1212; // -0.148 * 8192
static const int B2V422 =  -582; // -0.071 * 8192
static const int G2V422 = -3015; // -0.368 * 8192

static inline vuint8m1_t BGR2Y(int vl, const vuint8m1_t b, const vuint8m1_t g, const vuint8m1_t r)
{
    auto bb = __riscv_vzext_vf4(b, vl);
    auto gg = __riscv_vzext_vf4(g, vl);
    auto rr = __riscv_vzext_vf4(r, vl);
    auto yy = __riscv_vmadd(bb, B2Y422, __riscv_vmadd(gg, G2Y422, __riscv_vmadd(rr, R2Y422, __riscv_vmv_v_x_u32m4((16 << BGR2YUV422_SHIFT) + (1 << (BGR2YUV422_SHIFT - 1)), vl), vl), vl), vl);
    return __riscv_vnclipu(__riscv_vnclipu(yy, BGR2YUV422_SHIFT, __RISCV_VXRM_RDN, vl), 0, __RISCV_VXRM_RDN, vl);
}

static inline void BGR2UV(int vl, const vuint8m1_t b0, const vuint8m1_t g0, const vuint8m1_t r0,
                          const vuint8m1_t b1, const vuint8m1_t g1, const vuint8m1_t r1,
                          vuint8m1_t& u, vuint8m1_t& v)
{
    auto bb = __riscv_vadd(__riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4(b0, vl)), __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4(b1, vl)), vl);
    auto gg = __riscv_vadd(__riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4(g0, vl)), __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4(g1, vl)), vl);
    auto rr = __riscv_vadd(__riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4(r0, vl)), __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4(r1, vl)), vl);
    auto uu = __riscv_vmadd(bb, B2U422, __riscv_vmadd(gg, G2U422, __riscv_vmadd(rr, R2U422, __riscv_vmv_v_x_i32m4(257 << (BGR2YUV422_SHIFT - 1), vl), vl), vl), vl);
    auto vv = __riscv_vmadd(bb, B2V422, __riscv_vmadd(gg, G2V422, __riscv_vmadd(rr, B2U422, __riscv_vmv_v_x_i32m4(257 << (BGR2YUV422_SHIFT - 1), vl), vl), vl), vl);
    u = __riscv_vnclipu(__riscv_vreinterpret_v_i16m2_u16m2(__riscv_vmax(__riscv_vnclip(uu, BGR2YUV422_SHIFT, __RISCV_VXRM_RDN, vl), 0, vl)), 0, __RISCV_VXRM_RDN, vl);
    v = __riscv_vnclipu(__riscv_vreinterpret_v_i16m2_u16m2(__riscv_vmax(__riscv_vnclip(vv, BGR2YUV422_SHIFT, __RISCV_VXRM_RDN, vl), 0, vl)), 0, __RISCV_VXRM_RDN, vl);
}

static inline void cvtBGR82Yuv422(int vl, const vuint8m1_t b0, const vuint8m1_t g0, const vuint8m1_t r0,
                                  const vuint8m1_t b1, const vuint8m1_t g1, const vuint8m1_t r1,
                                  uchar* row, int yidx, int uidx, int vidx)
{
    auto vset = [](vuint8m1x4_t x, size_t p, vuint8m1_t y) {
        switch (p)
        {
        case 0:
            return __riscv_vset_v_u8m1_u8m1x4(x, 0, y);
        case 1:
            return __riscv_vset_v_u8m1_u8m1x4(x, 1, y);
        case 2:
            return __riscv_vset_v_u8m1_u8m1x4(x, 2, y);
        case 3:
            return __riscv_vset_v_u8m1_u8m1x4(x, 3, y);
        }
        /*NOTREACHED*/
    };

    vuint8m1_t u, v;
    BGR2UV(vl, b0, g0, r0, b1, g1, r1, u, v);

    vuint8m1x4_t x{};
    x = vset(x, uidx, u);
    x = vset(x, vidx, v);
    x = vset(x, yidx    , BGR2Y(vl, b0, g0, r0));
    x = vset(x, yidx + 2, BGR2Y(vl, b1, g1, r1));
    __riscv_vsseg4e8(row, x, vl);
}

// the algorithm is copied from imgproc/src/color_yuv.simd.cpp,
// in the functor struct RGB8toYUV422Invoker
static inline int cvtBGRtoSinglePlaneYUV(int start, int end, uchar * dst_data, size_t dst_step, int width, size_t stride, const uchar* src_data, int scn, bool swapBlue, int uIdx, int yIdx)
{
    // [yIdx, uIdx] | [uidx, vidx]:
    //     0, 0     |     1, 3
    //     0, 1     |     3, 1
    //     1, 0     |     0, 2
    const int uidx = 1 - yIdx + uIdx * 2;
    const int vidx = (2 + uidx) % 4;
    const uchar* bgr_src = src_data + start * stride;

    for (int j = start; j < end; j++, bgr_src += stride)
    {
        uchar* row = dst_data + dst_step * j;
        int vl;
        for (int i = 0; i < width / 2; i += vl)
        {
            vl = __riscv_vsetvl_e8m1(width / 2 - i);
            vuint8m1_t b0, g0, r0;
            vuint8m1_t b1, g1, r1;
            if (scn == 3)
            {
                auto x = __riscv_vlseg6e8_v_u8m1x6(bgr_src + 6 * i, vl);
                b0 = __riscv_vget_v_u8m1x6_u8m1(x, 0);
                g0 = __riscv_vget_v_u8m1x6_u8m1(x, 1);
                r0 = __riscv_vget_v_u8m1x6_u8m1(x, 2);
                b1 = __riscv_vget_v_u8m1x6_u8m1(x, 3);
                g1 = __riscv_vget_v_u8m1x6_u8m1(x, 4);
                r1 = __riscv_vget_v_u8m1x6_u8m1(x, 5);
            }
            else
            {
                auto x = __riscv_vlseg8e8_v_u8m1x8(bgr_src + 8 * i, vl);
                b0 = __riscv_vget_v_u8m1x8_u8m1(x, 0);
                g0 = __riscv_vget_v_u8m1x8_u8m1(x, 1);
                r0 = __riscv_vget_v_u8m1x8_u8m1(x, 2);
                b1 = __riscv_vget_v_u8m1x8_u8m1(x, 4);
                g1 = __riscv_vget_v_u8m1x8_u8m1(x, 5);
                r1 = __riscv_vget_v_u8m1x8_u8m1(x, 6);
            }
            if (swapBlue)
            {
                auto t = b0;
                b0 = r0, r0 = t;
                t = b1, b1 = r1, r1 = t;
            }

            cvtBGR82Yuv422(vl, b0, g0, r0, b1, g1, r1, row + 4 * i, yIdx, uidx, vidx);
        }
    }

    return CV_HAL_ERROR_OK;
}

// the algorithm is copied from imgproc/src/color_yuv.simd.cpp,
// in the functor struct RGB8toYUV420pInvoker
static inline int cvtBGRtoMultiPlaneYUV(int start, int end, uchar * yData, uchar * uvData, size_t dst_step, int width, int height, size_t stride, const uchar* src_data, int scn, bool swapBlue, int uIdx)
{
    uchar* yRow = (uchar*)0, *uRow = (uchar*)0, *vRow = (uchar*)0, *uvRow = (uchar*)0;
    for (int sRow = start*2; sRow < end*2; sRow++)
    {
        const uchar* srcRow = src_data + stride*sRow;
        yRow = yData + dst_step * sRow;
        bool evenRow = (sRow % 2) == 0;
        if (evenRow)
        {
            if (uIdx < 2)
            {
                uvRow = uvData + dst_step*(sRow/2);
            }
            else
            {
                uRow = uvData + dst_step * (sRow/4) + ((sRow/2) % 2) * (width/2);
                vRow = uvData + dst_step * ((sRow + height)/4) + (((sRow + height)/2) % 2) * (width/2);
            }
        }

        int vl;
        for (int i = 0; i < width / 2; i += vl)
        {
            vl = __riscv_vsetvl_e8m1(width / 2 - i);
            vuint8m1_t b0, g0, r0;
            vuint8m1_t b1, g1, r1;
            if (scn == 3)
            {
                auto x = __riscv_vlseg6e8_v_u8m1x6(srcRow + 6 * i, vl);
                b0 = __riscv_vget_v_u8m1x6_u8m1(x, 0);
                g0 = __riscv_vget_v_u8m1x6_u8m1(x, 1);
                r0 = __riscv_vget_v_u8m1x6_u8m1(x, 2);
                b1 = __riscv_vget_v_u8m1x6_u8m1(x, 3);
                g1 = __riscv_vget_v_u8m1x6_u8m1(x, 4);
                r1 = __riscv_vget_v_u8m1x6_u8m1(x, 5);
            }
            else
            {
                auto x = __riscv_vlseg8e8_v_u8m1x8(srcRow + 8 * i, vl);
                b0 = __riscv_vget_v_u8m1x8_u8m1(x, 0);
                g0 = __riscv_vget_v_u8m1x8_u8m1(x, 1);
                r0 = __riscv_vget_v_u8m1x8_u8m1(x, 2);
                b1 = __riscv_vget_v_u8m1x8_u8m1(x, 4);
                g1 = __riscv_vget_v_u8m1x8_u8m1(x, 5);
                r1 = __riscv_vget_v_u8m1x8_u8m1(x, 6);
            }
            if (swapBlue)
            {
                auto t = b0;
                b0 = r0, r0 = t;
                t = b1, b1 = r1, r1 = t;
            }

            auto y0 = bgrToY42x(vl, b0, g0, r0);
            auto y1 = bgrToY42x(vl, b1, g1, r1);
            __riscv_vsseg2e8(yRow + 2 * i, __riscv_vset_v_u8m1_u8m1x2(__riscv_vset_v_u8m1_u8m1x2(vuint8m1x2_t(), 0, y0), 1, y1), vl);

            if(evenRow)
            {
                vuint8m1_t uu, vv;
                bgrToUV42x(vl, b0, g0, r0, uu, vv);
                if(uIdx & 1)
                {
                    auto t = uu;
                    uu = vv, vv = t;
                }

                if(uIdx < 2)
                {
                    __riscv_vsseg2e8(uvRow + 2 * i, __riscv_vset_v_u8m1_u8m1x2(__riscv_vset_v_u8m1_u8m1x2(vuint8m1x2_t(), 0, uu), 1, vv), vl);
                }
                else
                {
                    __riscv_vse8(uRow + i, uu, vl);
                    __riscv_vse8(vRow + i, vv, vl);
                }
            }
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int cvtOnePlaneBGRtoYUV(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int scn, bool swapBlue, int uIdx, int yIdx)
{
    if (scn != 3 && scn != 4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    return color::invoke(height, -1, {cvtBGRtoSinglePlaneYUV}, dst_data, dst_step, width, src_step, src_data, scn, swapBlue, uIdx, yIdx);
}

inline int cvtBGRtoTwoPlaneYUV(const uchar * src_data, size_t src_step,
                               uchar * y_data, size_t y_step, uchar * uv_data, size_t uv_step,
                               int width, int height,
                               int scn, bool swapBlue, int uIdx)
{
    if (y_step != uv_step || (scn != 3 && scn != 4))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    return color::invoke(height / 2, -1, {cvtBGRtoMultiPlaneYUV}, y_data, uv_data, y_step, width, height, src_step, src_data, scn, swapBlue, uIdx == 2);
}

inline int cvtBGRtoThreePlaneYUV(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int scn, bool swapBlue, int uIdx)
{
    if (scn != 3 && scn != 4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    uchar* uv_data = dst_data + dst_step * static_cast<size_t>(height);
    return color::invoke(height / 2, -1, {cvtBGRtoMultiPlaneYUV}, dst_data, uv_data, dst_step, width, height, src_step, src_data, scn, swapBlue, uIdx == 2 ? 3 : 2);
}
} // cv::cv_hal_rvv::PlaneBGRtoYUV

namespace HSVtoBGR {
#undef cv_hal_cvtHSVtoBGR
#define cv_hal_cvtHSVtoBGR cv::cv_hal_rvv::HSVtoBGR::cvtHSVtoBGR

template<typename T>
static inline int cvtHSVtoBGR(int start, int end, const T * src, size_t src_step, T * dst, size_t dst_step, int width, int dcn, bool swapBlue, bool isFullRange, bool isHSV);

static inline void ComputeSectorAndClampedH(int vl, vfloat32m2_t& h, vint32m2_t& sector)
{
    int rd;
    asm volatile("fsrmi %0, %1" : "=r"(rd) : "i"(2)); // Rounding Mode: RDN
    sector = __riscv_vfcvt_x(h, vl);
    asm volatile("fsrm %0" : : "r"(rd));

    h = __riscv_vfsub(h, __riscv_vfcvt_f(sector, vl), vl);
    sector = __riscv_vrem(sector, 6, vl);
    sector = __riscv_vadd_mu(__riscv_vmslt(sector, 0, vl), sector, sector, 6, vl);
}

static inline void Hxx2BGR_loadtab(int vl, vfloat32m2_t tab0, vfloat32m2_t tab1, vfloat32m2_t tab2, vfloat32m2_t tab3,
                                   vint32m2_t sector, vfloat32m2_t& b, vfloat32m2_t& g, vfloat32m2_t& r)
{
    static const uint sector_data[3][6] =
    {
        {1, 1, 3, 0, 0, 2},
        {3, 0, 0, 2, 1, 1},
        {0, 2, 1, 1, 3, 0}
    };
    auto loadtab = [&](size_t p) {
        auto sd = __riscv_vloxei32_v_u32m2(sector_data[p], __riscv_vreinterpret_v_i32m2_u32m2(sector), vl);
        auto x = __riscv_vmerge(vfloat32m2_t(), tab0, __riscv_vmseq(sd, 0, vl), vl);
        x = __riscv_vmerge(x, tab1, __riscv_vmseq(sd, 1, vl), vl);
        x = __riscv_vmerge(x, tab2, __riscv_vmseq(sd, 2, vl), vl);
        return __riscv_vmerge(x, tab3, __riscv_vmseq(sd, 3, vl), vl);
    };

    sector = __riscv_vmul(sector, sizeof(uint), vl);
    b = loadtab(0);
    g = loadtab(1);
    r = loadtab(2);
}

static inline void HSV2BGR_native(int vl, vfloat32m2_t h, vfloat32m2_t s, vfloat32m2_t v,
                                  vfloat32m2_t& b, vfloat32m2_t& g, vfloat32m2_t& r,
                                  const float hscale)
{
    h = __riscv_vfmul(h, hscale, vl);
    vint32m2_t sector;
    ComputeSectorAndClampedH(vl, h, sector);

    auto tab0 = v;
    auto tab1 = __riscv_vfnmsub(v, s, v, vl);
    auto tab2 = __riscv_vfnmsub(__riscv_vfmul(v, s, vl), h, v, vl);
    auto tab3 = __riscv_vfadd(v, __riscv_vfsub(tab1, tab2, vl), vl);
    Hxx2BGR_loadtab(vl, tab0, tab1, tab2, tab3, sector, b, g, r);
}

static inline void HLS2BGR_native(int vl, vfloat32m2_t h, vfloat32m2_t l, vfloat32m2_t s,
                                  vfloat32m2_t& b, vfloat32m2_t& g, vfloat32m2_t& r,
                                  const float hscale)
{
    h = __riscv_vfmul(h, hscale, vl);
    vint32m2_t sector;
    ComputeSectorAndClampedH(vl, h, sector);

    auto tab0 = __riscv_vmerge(__riscv_vfnmsub(l, s, __riscv_vfadd(l, s, vl), vl), __riscv_vfmadd(l, s, l, vl), __riscv_vmfle(l, 0.5f, vl), vl);
    auto tab1 = __riscv_vfsub(__riscv_vfadd(l, l, vl), tab0, vl);
    auto tab3 = __riscv_vfmadd(__riscv_vfsub(tab0, tab1, vl), h, tab1, vl);
    auto tab2 = __riscv_vfsub(__riscv_vfadd(tab0, tab1, vl), tab3, vl);
    Hxx2BGR_loadtab(vl, tab0, tab1, tab2, tab3, sector, b, g, r);
}

// the algorithm is copied from imgproc/src/color_hsv.simd.cpp,
// in the functor struct HSV2RGB_f, HSV2RGB_b, HLS2RGB_f and HLS2RGB_b
template<>
inline int cvtHSVtoBGR<uchar>(int start, int end, const uchar * src, size_t src_step, uchar * dst, size_t dst_step, int width, int dcn, bool swapBlue, bool isFullRange, bool isHSV)
{
    float hs = 6.0f / (isFullRange ? 255 : 180), r255 = 1.0f / 255;
    auto alpha = __riscv_vmv_v_x_u8mf2(std::numeric_limits<uchar>::max(), __riscv_vsetvlmax_e8mf2());
    for (int i = start; i < end; i++)
    {
        int vl;
        for (int j = 0; j < width; j += vl)
        {
            vl = __riscv_vsetvl_e8mf2(width - j);
            auto x = __riscv_vlseg3e8_v_u8mf2x3(src + i * src_step + j * 3, vl);
            auto h = __riscv_vfwcvt_f(__riscv_vwcvtu_x(__riscv_vget_v_u8mf2x3_u8mf2(x, 0), vl), vl);
            auto s = __riscv_vfmul(__riscv_vfwcvt_f(__riscv_vwcvtu_x(__riscv_vget_v_u8mf2x3_u8mf2(x, 1), vl), vl), r255, vl);
            auto v = __riscv_vfmul(__riscv_vfwcvt_f(__riscv_vwcvtu_x(__riscv_vget_v_u8mf2x3_u8mf2(x, 2), vl), vl), r255, vl);

            vfloat32m2_t b, g, r;
            isHSV ? HSV2BGR_native(vl, h, s, v, b, g, r, hs) : HLS2BGR_native(vl, h, s, v, b, g, r, hs);
            if (swapBlue)
            {
                auto t = b;
                b = r, r = t;
            }
            b = __riscv_vfmul(b, 255.0f, vl);
            g = __riscv_vfmul(g, 255.0f, vl);
            r = __riscv_vfmul(r, 255.0f, vl);

            if (dcn == 3)
            {
                vuint8mf2x3_t y{};
                y = __riscv_vset_v_u8mf2_u8mf2x3(y, 0, __riscv_vnclipu(__riscv_vfncvt_xu(b, vl), 0, __RISCV_VXRM_RNU, vl));
                y = __riscv_vset_v_u8mf2_u8mf2x3(y, 1, __riscv_vnclipu(__riscv_vfncvt_xu(g, vl), 0, __RISCV_VXRM_RNU, vl));
                y = __riscv_vset_v_u8mf2_u8mf2x3(y, 2, __riscv_vnclipu(__riscv_vfncvt_xu(r, vl), 0, __RISCV_VXRM_RNU, vl));
                __riscv_vsseg3e8(dst + i * dst_step + j * 3, y, vl);
            }
            else
            {
                vuint8mf2x4_t y{};
                y = __riscv_vset_v_u8mf2_u8mf2x4(y, 0, __riscv_vnclipu(__riscv_vfncvt_xu(b, vl), 0, __RISCV_VXRM_RNU, vl));
                y = __riscv_vset_v_u8mf2_u8mf2x4(y, 1, __riscv_vnclipu(__riscv_vfncvt_xu(g, vl), 0, __RISCV_VXRM_RNU, vl));
                y = __riscv_vset_v_u8mf2_u8mf2x4(y, 2, __riscv_vnclipu(__riscv_vfncvt_xu(r, vl), 0, __RISCV_VXRM_RNU, vl));
                y = __riscv_vset_v_u8mf2_u8mf2x4(y, 3, alpha);
                __riscv_vsseg4e8(dst + i * dst_step + j * 4, y, vl);
            }
        }
    }

    return CV_HAL_ERROR_OK;
}

template<>
inline int cvtHSVtoBGR<float>(int start, int end, const float * src, size_t src_step, float * dst, size_t dst_step, int width, int dcn, bool swapBlue, bool isFullRange, bool isHSV)
{
    src_step /= sizeof(float);
    dst_step /= sizeof(float);

    float hs = 6.0f / 360;
    auto alpha = __riscv_vfmv_v_f_f32m2(1.0f, __riscv_vsetvlmax_e32m2());
    for (int i = start; i < end; i++)
    {
        int vl;
        for (int j = 0; j < width; j += vl)
        {
            vl = __riscv_vsetvl_e32m2(width - j);
            auto x = __riscv_vlseg3e32_v_f32m2x3(src + i * src_step + j * 3, vl);
            auto h = __riscv_vget_v_f32m2x3_f32m2(x, 0), s = __riscv_vget_v_f32m2x3_f32m2(x, 1), v = __riscv_vget_v_f32m2x3_f32m2(x, 2);

            vfloat32m2_t b, g, r;
            isHSV ? HSV2BGR_native(vl, h, s, v, b, g, r, hs) : HLS2BGR_native(vl, h, s, v, b, g, r, hs);
            if (swapBlue)
            {
                auto t = b;
                b = r, r = t;
            }

            if (dcn == 3)
            {
                vfloat32m2x3_t y{};
                y = __riscv_vset_v_f32m2_f32m2x3(y, 0, b);
                y = __riscv_vset_v_f32m2_f32m2x3(y, 1, g);
                y = __riscv_vset_v_f32m2_f32m2x3(y, 2, r);
                __riscv_vsseg3e32(dst + i * dst_step + j * 3, y, vl);
            }
            else
            {
                vfloat32m2x4_t y{};
                y = __riscv_vset_v_f32m2_f32m2x4(y, 0, b);
                y = __riscv_vset_v_f32m2_f32m2x4(y, 1, g);
                y = __riscv_vset_v_f32m2_f32m2x4(y, 2, r);
                y = __riscv_vset_v_f32m2_f32m2x4(y, 3, alpha);
                __riscv_vsseg4e32(dst + i * dst_step + j * 4, y, vl);
            }
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int cvtHSVtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int depth, int dcn, bool swapBlue, bool isFullRange, bool isHSV)
{
    if (dcn != 3 && dcn != 4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    switch (depth)
    {
    case CV_8U:
        return color::invoke(height, -1, cvtHSVtoBGR<uchar>, reinterpret_cast<const uchar*>(src_data), src_step, reinterpret_cast<uchar*>(dst_data), dst_step, width, dcn, swapBlue, isFullRange, isHSV);
    case CV_32F:
        return color::invoke(height, -1, cvtHSVtoBGR<float>, reinterpret_cast<const float*>(src_data), src_step, reinterpret_cast<float*>(dst_data), dst_step, width, dcn, swapBlue, isFullRange, isHSV);
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}
} // cv::cv_hal_rvv::HSVtoBGR

namespace BGRtoHSV {
#undef cv_hal_cvtBGRtoHSV
#define cv_hal_cvtBGRtoHSV cv::cv_hal_rvv::BGRtoHSV::cvtBGRtoHSV

template<typename T>
static inline int cvtBGRtoHSV(int start, int end, const T * src, size_t src_step, T * dst, size_t dst_step, int width, int scn, bool swapBlue, bool isFullRange, bool isHSV);

// the algorithm is copied from imgproc/src/color_hsv.simd.cpp,
// in the functor struct RGB2HSV_f, RGB2HSV_b, RGB2HLS_f and RGB2HLS_b
template<>
inline int cvtBGRtoHSV<uchar>(int start, int end, const uchar * src, size_t src_step, uchar * dst, size_t dst_step, int width, int scn, bool swapBlue, bool isFullRange, bool isHSV)
{
    for (int i = start; i < end; i++)
    {
        int vl;
        for (int j = 0; j < width; j += vl)
        {
            vl = __riscv_vsetvl_e8mf2(width - j);
            vint32m2_t b, g, r;
            if (scn == 3)
            {
                auto x = __riscv_vlseg3e8_v_u8mf2x3(src + i * src_step + j * 3, vl);
                b = __riscv_vreinterpret_v_u32m2_i32m2(__riscv_vzext_vf4(__riscv_vget_v_u8mf2x3_u8mf2(x, 0), vl));
                g = __riscv_vreinterpret_v_u32m2_i32m2(__riscv_vzext_vf4(__riscv_vget_v_u8mf2x3_u8mf2(x, 1), vl));
                r = __riscv_vreinterpret_v_u32m2_i32m2(__riscv_vzext_vf4(__riscv_vget_v_u8mf2x3_u8mf2(x, 2), vl));
            }
            else
            {
                auto x = __riscv_vlseg4e8_v_u8mf2x4(src + i * src_step + j * 4, vl);
                b = __riscv_vreinterpret_v_u32m2_i32m2(__riscv_vzext_vf4(__riscv_vget_v_u8mf2x4_u8mf2(x, 0), vl));
                g = __riscv_vreinterpret_v_u32m2_i32m2(__riscv_vzext_vf4(__riscv_vget_v_u8mf2x4_u8mf2(x, 1), vl));
                r = __riscv_vreinterpret_v_u32m2_i32m2(__riscv_vzext_vf4(__riscv_vget_v_u8mf2x4_u8mf2(x, 2), vl));
            }
            if (swapBlue)
            {
                auto t = b;
                b = r, r = t;
            }

            auto v = b, vmin = b;
            v = __riscv_vmax(v, g, vl);
            v = __riscv_vmax(v, r, vl);
            vmin = __riscv_vmin(vmin, g, vl);
            vmin = __riscv_vmin(vmin, r, vl);
            auto diff = __riscv_vsub(v, vmin, vl);

            auto s = __riscv_vnclip(__riscv_vmul(diff, __riscv_vfcvt_x(__riscv_vfrdiv(__riscv_vfcvt_f(v, vl), 255 << 12, vl), vl), vl), 12, __RISCV_VXRM_RNU, vl);
            auto h = __riscv_vmadd(diff, 4, __riscv_vsub(r, g, vl), vl);
            h = __riscv_vmerge(h, __riscv_vmadd(diff, 2, __riscv_vsub(b, r, vl), vl), __riscv_vmseq(v, g, vl), vl);
            h = __riscv_vmerge(h, __riscv_vsub(g, b, vl), __riscv_vmseq(v, r, vl), vl);
            h = __riscv_vssra(__riscv_vmul(h, __riscv_vfcvt_x(__riscv_vfrdiv(__riscv_vfcvt_f(__riscv_vmul(diff, 6, vl), vl), isFullRange ? 256 << 12 : 180 << 12, vl), vl), vl), 12, __RISCV_VXRM_RNU, vl);
            h = __riscv_vadd_mu(__riscv_vmslt(h, 0, vl), h, h, isFullRange ? 256 : 180, vl);
            // auto s = __riscv_vdiv(diff, __riscv_vadd(__riscv_vabs(v, vl), FLT_EPSILON, vl), vl);
            // diff = __riscv_vrdiv(__riscv_vadd(diff, FLT_EPSILON, vl), 60.0f, vl);

            // auto h = __riscv_vmadd(__riscv_vsub(r, g, vl), diff, __riscv_vmv_v_f_f32m2(240.0f, vl), vl);
            // h = __riscv_vmerge(h, __riscv_vmadd(__riscv_vsub(b, r, vl), diff, __riscv_vmv_v_f_f32m2(120.0f, vl), vl), __riscv_vmfeq(v, g, vl), vl);
            // h = __riscv_vmerge(h, __riscv_vmul(__riscv_vsub(g, b, vl), diff, vl), __riscv_vmfeq(v, r, vl), vl);
            // h = __riscv_vadd_mu(__riscv_vmflt(h, 0, vl), h, h, 360.0f, vl);

            vuint8mf2x3_t x{};
            x = __riscv_vset_v_u8mf2_u8mf2x3(x, 0, __riscv_vnclipu(__riscv_vnclipu(__riscv_vreinterpret_v_i32m2_u32m2(h), 0, __RISCV_VXRM_RNU, vl), 0, __RISCV_VXRM_RNU, vl));
            x = __riscv_vset_v_u8mf2_u8mf2x3(x, 1, __riscv_vncvt_x(__riscv_vreinterpret_v_i16m1_u16m1(s), vl));
            x = __riscv_vset_v_u8mf2_u8mf2x3(x, 2, __riscv_vncvt_x(__riscv_vncvt_x(__riscv_vreinterpret_v_i32m2_u32m2(v), vl), vl));
            __riscv_vsseg3e8(dst + i * dst_step + j * 3, x, vl);
        }
    }

    return CV_HAL_ERROR_OK;
}

template<>
inline int cvtBGRtoHSV<float>(int start, int end, const float * src, size_t src_step, float * dst, size_t dst_step, int width, int scn, bool swapBlue, bool isFullRange, bool isHSV)
{
    src_step /= sizeof(float);
    dst_step /= sizeof(float);

    for (int i = start; i < end; i++)
    {
        int vl;
        for (int j = 0; j < width; j += vl)
        {
            vl = __riscv_vsetvl_e32m2(width - j);
            vfloat32m2_t b, g, r;
            if (scn == 3)
            {
                auto x = __riscv_vlseg3e32_v_f32m2x3(src + i * src_step + j * 3, vl);
                b = __riscv_vget_v_f32m2x3_f32m2(x, 0);
                g = __riscv_vget_v_f32m2x3_f32m2(x, 1);
                r = __riscv_vget_v_f32m2x3_f32m2(x, 2);
            }
            else
            {
                auto x = __riscv_vlseg4e32_v_f32m2x4(src + i * src_step + j * 4, vl);
                b = __riscv_vget_v_f32m2x4_f32m2(x, 0);
                g = __riscv_vget_v_f32m2x4_f32m2(x, 1);
                r = __riscv_vget_v_f32m2x4_f32m2(x, 2);
            }
            if (swapBlue)
            {
                auto t = b;
                b = r, r = t;
            }

            auto v = b, vmin = b;
            v = __riscv_vfmax(v, g, vl);
            v = __riscv_vfmax(v, r, vl);
            vmin = __riscv_vfmin(vmin, g, vl);
            vmin = __riscv_vfmin(vmin, r, vl);
            auto diff = __riscv_vfsub(v, vmin, vl);
            auto s = __riscv_vfdiv(diff, __riscv_vfadd(__riscv_vfabs(v, vl), FLT_EPSILON, vl), vl);
            diff = __riscv_vfrdiv(__riscv_vfadd(diff, FLT_EPSILON, vl), 60.0f, vl);

            auto h = __riscv_vfmadd(__riscv_vfsub(r, g, vl), diff, __riscv_vfmv_v_f_f32m2(240.0f, vl), vl);
            h = __riscv_vmerge(h, __riscv_vfmadd(__riscv_vfsub(b, r, vl), diff, __riscv_vfmv_v_f_f32m2(120.0f, vl), vl), __riscv_vmfeq(v, g, vl), vl);
            h = __riscv_vmerge(h, __riscv_vfmul(__riscv_vfsub(g, b, vl), diff, vl), __riscv_vmfeq(v, r, vl), vl);
            h = __riscv_vfadd_mu(__riscv_vmflt(h, 0, vl), h, h, 360.0f, vl);

            vfloat32m2x3_t x{};
            x = __riscv_vset_v_f32m2_f32m2x3(x, 0, h);
            x = __riscv_vset_v_f32m2_f32m2x3(x, 1, s);
            x = __riscv_vset_v_f32m2_f32m2x3(x, 2, v);
            __riscv_vsseg3e32(dst + i * dst_step + j * 3, x, vl);
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int cvtBGRtoHSV(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int depth, int scn, bool swapBlue, bool isFullRange, bool isHSV)
{
    if (scn != 3 && scn != 4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    switch (depth)
    {
    case CV_8U:
        return color::invoke(height, -1, cvtBGRtoHSV<uchar>, reinterpret_cast<const uchar*>(src_data), src_step, reinterpret_cast<uchar*>(dst_data), dst_step, width, scn, swapBlue, isFullRange, isHSV);
    case CV_32F:
        return color::invoke(height, -1, cvtBGRtoHSV<float>, reinterpret_cast<const float*>(src_data), src_step, reinterpret_cast<float*>(dst_data), dst_step, width, scn, swapBlue, isFullRange, isHSV);
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}
} // cv::cv_hal_rvv::BGRtoHSV

}}

#endif
