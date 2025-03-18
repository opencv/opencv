// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#ifndef OPENCV_HAL_RVV_COLOR_HPP_INCLUDED
#define OPENCV_HAL_RVV_COLOR_HPP_INCLUDED

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv {

namespace color {
    class ColorInvoker : public ParallelLoopBody
    {
    public:
        template<typename... Args>
        ColorInvoker(std::function<int(int, int, Args...)> _func, Args&&... args)
        {
            func = std::bind(_func, std::placeholders::_1, std::placeholders::_2, std::forward<Args>(args)...);
        }

        virtual void operator()(const Range& range) const override
        {
            func(range.start, range.end);
        }

    private:
        std::function<int(int, int)> func;
    };

    template<typename... Args>
    static inline int invoke(int width, int height, std::function<int(int, int, Args...)> func, Args&&... args)
    {
        cv::parallel_for_(Range(1, height), ColorInvoker(func, std::forward<Args>(args)...), (width - 1) * height / static_cast<double>(1 << 15));
        return func(0, 1, std::forward<Args>(args)...);
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
    static inline void vsseg(uchar* a, int b, T c, T d, T e, T f, size_t g)
    {
        if (b == 3)
        {
            vuint8m2x3_t x{};
            x = __riscv_vset_v_u8m2_u8m2x3(x, 0, c);
            x = __riscv_vset_v_u8m2_u8m2x3(x, 1, d);
            x = __riscv_vset_v_u8m2_u8m2x3(x, 2, e);
            __riscv_vsseg3e8(a, x, g);
        }
        else
        {
            vuint8m2x4_t x{};
            x = __riscv_vset_v_u8m2_u8m2x4(x, 0, c);
            x = __riscv_vset_v_u8m2_u8m2x4(x, 1, d);
            x = __riscv_vset_v_u8m2_u8m2x4(x, 2, e);
            x = __riscv_vset_v_u8m2_u8m2x4(x, 3, f);
            __riscv_vsseg4e8(a, x, g);
        }
    }
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
    static inline void vsseg(ushort* a, int b, T c, T d, T e, T f, size_t g)
    {
        if (b == 3)
        {
            vuint16m2x3_t x{};
            x = __riscv_vset_v_u16m2_u16m2x3(x, 0, c);
            x = __riscv_vset_v_u16m2_u16m2x3(x, 1, d);
            x = __riscv_vset_v_u16m2_u16m2x3(x, 2, e);
            __riscv_vsseg3e16(a, x, g);
        }
        else
        {
            vuint16m2x4_t x{};
            x = __riscv_vset_v_u16m2_u16m2x4(x, 0, c);
            x = __riscv_vset_v_u16m2_u16m2x4(x, 1, d);
            x = __riscv_vset_v_u16m2_u16m2x4(x, 2, e);
            x = __riscv_vset_v_u16m2_u16m2x4(x, 3, f);
            __riscv_vsseg4e16(a, x, g);
        }
    }
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
    static inline void vsseg(float* a, int b, T c, T d, T e, T f, size_t g)
    {
        if (b == 3)
        {
            vfloat32m2x3_t x{};
            x = __riscv_vset_v_f32m2_f32m2x3(x, 0, c);
            x = __riscv_vset_v_f32m2_f32m2x3(x, 1, d);
            x = __riscv_vset_v_f32m2_f32m2x3(x, 2, e);
            __riscv_vsseg3e32(a, x, g);
        }
        else
        {
            vfloat32m2x4_t x{};
            x = __riscv_vset_v_f32m2_f32m2x4(x, 0, c);
            x = __riscv_vset_v_f32m2_f32m2x4(x, 1, d);
            x = __riscv_vset_v_f32m2_f32m2x4(x, 2, e);
            x = __riscv_vset_v_f32m2_f32m2x4(x, 3, f);
            __riscv_vsseg4e32(a, x, g);
        }
    }
    static inline T vmv_v_x(float a, size_t b) { return __riscv_vfmv_v_f_f32m2(a, b); }
};

// the algorithm is copied from imgproc/src/color_rgb.simd.cpp,
// in the functor struct RGB2RGB
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
        return cvtBGRtoBGR<uchar>(0, height, reinterpret_cast<const uchar*>(src_data), src_step, reinterpret_cast<uchar*>(dst_data), dst_step, width, scn, dcn, swapBlue);
    case CV_16U:
        return cvtBGRtoBGR<ushort>(0, height, reinterpret_cast<const ushort*>(src_data), src_step, reinterpret_cast<ushort*>(dst_data), dst_step, width, scn, dcn, swapBlue);
    case CV_32F:
        return cvtBGRtoBGR<float>(0, height, reinterpret_cast<const float*>(src_data), src_step, reinterpret_cast<float*>(dst_data), dst_step, width, scn, dcn, swapBlue);
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
    static inline void vsseg(uchar* a, int b, T c, T d, T e, T f, size_t g)
    {
        if (b == 3)
        {
            vuint8m2x3_t x{};
            x = __riscv_vset_v_u8m2_u8m2x3(x, 0, c);
            x = __riscv_vset_v_u8m2_u8m2x3(x, 1, d);
            x = __riscv_vset_v_u8m2_u8m2x3(x, 2, e);
            __riscv_vsseg3e8(a, x, g);
        }
        else
        {
            vuint8m2x4_t x{};
            x = __riscv_vset_v_u8m2_u8m2x4(x, 0, c);
            x = __riscv_vset_v_u8m2_u8m2x4(x, 1, d);
            x = __riscv_vset_v_u8m2_u8m2x4(x, 2, e);
            x = __riscv_vset_v_u8m2_u8m2x4(x, 3, f);
            __riscv_vsseg4e8(a, x, g);
        }
    }
    static inline T vmv_v_x(uchar a, size_t b) { return __riscv_vmv_v_x_u8m2(a, b); }
};
template<> struct rvv<ushort>
{
    using T = vuint16m2_t;
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e16m2(); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e16m2(a); }
    static inline T vle(const ushort* a, size_t b) { return __riscv_vle16_v_u16m2(a, b); }
    static inline void vsseg(ushort* a, int b, T c, T d, T e, T f, size_t g)
    {
        if (b == 3)
        {
            vuint16m2x3_t x{};
            x = __riscv_vset_v_u16m2_u16m2x3(x, 0, c);
            x = __riscv_vset_v_u16m2_u16m2x3(x, 1, d);
            x = __riscv_vset_v_u16m2_u16m2x3(x, 2, e);
            __riscv_vsseg3e16(a, x, g);
        }
        else
        {
            vuint16m2x4_t x{};
            x = __riscv_vset_v_u16m2_u16m2x4(x, 0, c);
            x = __riscv_vset_v_u16m2_u16m2x4(x, 1, d);
            x = __riscv_vset_v_u16m2_u16m2x4(x, 2, e);
            x = __riscv_vset_v_u16m2_u16m2x4(x, 3, f);
            __riscv_vsseg4e16(a, x, g);
        }
    }
    static inline T vmv_v_x(ushort a, size_t b) { return __riscv_vmv_v_x_u16m2(a, b); }
};
template<> struct rvv<float>
{
    using T = vfloat32m2_t;
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e32m2(); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e32m2(a); }
    static inline T vle(const float* a, size_t b) { return __riscv_vle32_v_f32m2(a, b); }
    static inline void vsseg(float* a, int b, T c, T d, T e, T f, size_t g)
    {
        if (b == 3)
        {
            vfloat32m2x3_t x{};
            x = __riscv_vset_v_f32m2_f32m2x3(x, 0, c);
            x = __riscv_vset_v_f32m2_f32m2x3(x, 1, d);
            x = __riscv_vset_v_f32m2_f32m2x3(x, 2, e);
            __riscv_vsseg3e32(a, x, g);
        }
        else
        {
            vfloat32m2x4_t x{};
            x = __riscv_vset_v_f32m2_f32m2x4(x, 0, c);
            x = __riscv_vset_v_f32m2_f32m2x4(x, 1, d);
            x = __riscv_vset_v_f32m2_f32m2x4(x, 2, e);
            x = __riscv_vset_v_f32m2_f32m2x4(x, 3, f);
            __riscv_vsseg4e32(a, x, g);
        }
    }
    static inline T vmv_v_x(float a, size_t b) { return __riscv_vfmv_v_f_f32m2(a, b); }
};

// the algorithm is copied from imgproc/src/color_rgb.simd.cpp,
// in the functor struct Gray2RGB
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
        return cvtGraytoBGR<uchar>(0, height, reinterpret_cast<const uchar*>(src_data), src_step, reinterpret_cast<uchar*>(dst_data), dst_step, width, dcn);
    case CV_16U:
        return cvtGraytoBGR<ushort>(0, height, reinterpret_cast<const ushort*>(src_data), src_step, reinterpret_cast<ushort*>(dst_data), dst_step, width, dcn);
    case CV_32F:
        return cvtGraytoBGR<float>(0, height, reinterpret_cast<const float*>(src_data), src_step, reinterpret_cast<float*>(dst_data), dst_step, width, dcn);
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
    static inline T vcvt1(vuint32m4_t a, size_t b, size_t c) { return __riscv_vnclipu(__riscv_vnclipu(a, b, __RISCV_VXRM_RNU, c), 0, __RISCV_VXRM_RNU, c); }
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
    static inline T vcvt1(vuint32m4_t a, size_t b, size_t c) { return __riscv_vnclipu(a, b, __RISCV_VXRM_RNU, c); }
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

// the algorithm is copied from imgproc/src/color_rgb.simd.cpp,
// in the functor struct RGB2Gray
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
        return color::invoke(width, height, {cvtBGRtoGray<uchar>}, reinterpret_cast<const uchar*>(src_data), src_step, reinterpret_cast<uchar*>(dst_data), dst_step, width, scn, swapBlue);
    case CV_16U:
        return color::invoke(width, height, {cvtBGRtoGray<ushort>}, reinterpret_cast<const ushort*>(src_data), src_step, reinterpret_cast<ushort*>(dst_data), dst_step, width, scn, swapBlue);
    case CV_32F:
        return color::invoke(width, height, {cvtBGRtoGray<float>}, reinterpret_cast<const float*>(src_data), src_step, reinterpret_cast<float*>(dst_data), dst_step, width, scn, swapBlue);
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}
} // cv::cv_hal_rvv::BGRtoGray

namespace BGR5x5toBGR {
#undef cv_hal_cvtBGR5x5toBGR
#define cv_hal_cvtBGR5x5toBGR cv::cv_hal_rvv::BGR5x5toBGR::cvtBGR5x5toBGR

// the algorithm is copied from imgproc/src/color_rgb.simd.cpp,
// in the functor struct RGB5x52RGB
static inline int cvtBGR5x5toBGR_u(int start, int end, const ushort * src, size_t src_step, uchar * dst, size_t dst_step, int width, int dcn, bool swapBlue, int greenBits)
{
    src_step /= sizeof(ushort);

    for (int i = start; i < end; i++)
    {
        int vl;
        for (int j = 0; j < width; j += vl)
        {
            vl = __riscv_vsetvl_e16m4(width - j);
            auto vec_src = __riscv_vle16_v_u16m4(src + i * src_step + j, vl);

            auto vec_dstB = __riscv_vncvt_x(__riscv_vsll(vec_src, 3, vl), vl), vec_dstA = __riscv_vmv_v_x_u8m2(std::numeric_limits<uchar>::max(), vl);
            vuint8m2_t vec_dstG, vec_dstR;
            if (greenBits == 6)
            {
                vec_dstG = __riscv_vncvt_x(__riscv_vand(__riscv_vsrl(vec_src, 3, vl), ~3, vl), vl);
                vec_dstR = __riscv_vncvt_x(__riscv_vand(__riscv_vsrl(vec_src, 8, vl), ~7, vl), vl);
            }
            else
            {
                vec_dstG = __riscv_vncvt_x(__riscv_vand(__riscv_vsrl(vec_src, 2, vl), ~7, vl), vl);
                vec_dstR = __riscv_vncvt_x(__riscv_vand(__riscv_vsrl(vec_src, 7, vl), ~7, vl), vl);
                vec_dstA = __riscv_vmerge(vec_dstA, 0, __riscv_vmsltu(vec_src, 0x8000, vl), vl);
            }
            if (swapBlue)
            {
                auto t = vec_dstB;
                vec_dstB = vec_dstR, vec_dstR = t;
            }

            if (dcn == 3)
            {
                vuint8m2x3_t x{};
                x = __riscv_vset_v_u8m2_u8m2x3(x, 0, vec_dstB);
                x = __riscv_vset_v_u8m2_u8m2x3(x, 1, vec_dstG);
                x = __riscv_vset_v_u8m2_u8m2x3(x, 2, vec_dstR);
                __riscv_vsseg3e8(dst + i * dst_step + j * 3, x, vl);
            }
            else
            {
                vuint8m2x4_t x{};
                x = __riscv_vset_v_u8m2_u8m2x4(x, 0, vec_dstB);
                x = __riscv_vset_v_u8m2_u8m2x4(x, 1, vec_dstG);
                x = __riscv_vset_v_u8m2_u8m2x4(x, 2, vec_dstR);
                x = __riscv_vset_v_u8m2_u8m2x4(x, 3, vec_dstA);
                __riscv_vsseg4e8(dst + i * dst_step + j * 4, x, vl);
            }
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int cvtBGR5x5toBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int dcn, bool swapBlue, int greenBits)
{
    if ((dcn != 3 && dcn != 4) || (greenBits != 5 && greenBits != 6))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    return color::invoke(width, height, {cvtBGR5x5toBGR_u}, reinterpret_cast<const ushort*>(src_data), src_step, reinterpret_cast<uchar*>(dst_data), dst_step, width, dcn, swapBlue, greenBits);
}
} // cv::cv_hal_rvv::BGR5x5toBGR

namespace BGRtoBGR5x5 {
#undef cv_hal_cvtBGRtoBGR5x5
#define cv_hal_cvtBGRtoBGR5x5 cv::cv_hal_rvv::BGRtoBGR5x5::cvtBGRtoBGR5x5

// the algorithm is copied from imgproc/src/color_rgb.simd.cpp,
// in the functor struct RGB2RGB5x5
static inline int cvtBGRtoBGR5x5_u(int start, int end, const uchar * src, size_t src_step, ushort * dst, size_t dst_step, int width, int scn, bool swapBlue, int greenBits)
{
    dst_step /= sizeof(ushort);

    for (int i = start; i < end; i++)
    {
        int vl;
        for (int j = 0; j < width; j += vl)
        {
            vl = __riscv_vsetvl_e8m2(width - j);
            vuint16m4_t vec_srcB, vec_srcG, vec_srcR, vec_srcA = __riscv_vmv_v_x_u16m4(0, vl);
            if (scn == 3)
            {
                auto x = __riscv_vlseg3e8_v_u8m2x3(src + i * src_step + j * 3, vl);
                vec_srcB = __riscv_vwcvtu_x(__riscv_vget_v_u8m2x3_u8m2(x, 0), vl);
                vec_srcG = __riscv_vwcvtu_x(__riscv_vget_v_u8m2x3_u8m2(x, 1), vl);
                vec_srcR = __riscv_vwcvtu_x(__riscv_vget_v_u8m2x3_u8m2(x, 2), vl);
            }
            else
            {
                auto x = __riscv_vlseg4e8_v_u8m2x4(src + i * src_step + j * 4, vl);
                vec_srcB = __riscv_vwcvtu_x(__riscv_vget_v_u8m2x4_u8m2(x, 0), vl);
                vec_srcG = __riscv_vwcvtu_x(__riscv_vget_v_u8m2x4_u8m2(x, 1), vl);
                vec_srcR = __riscv_vwcvtu_x(__riscv_vget_v_u8m2x4_u8m2(x, 2), vl);
                vec_srcA = __riscv_vwcvtu_x(__riscv_vget_v_u8m2x4_u8m2(x, 3), vl);
            }
            if (swapBlue)
            {
                auto t = vec_srcB;
                vec_srcB = vec_srcR, vec_srcR = t;
            }

            auto vec_dst = __riscv_vsrl(vec_srcB, 3, vl);
            if (greenBits == 6)
            {
                vec_dst = __riscv_vor(__riscv_vor(vec_dst, __riscv_vsll(__riscv_vand(vec_srcG, ~3, vl), 3, vl), vl), __riscv_vsll(__riscv_vand(vec_srcR, ~7, vl), 8, vl), vl);
            }
            else
            {
                vec_dst = __riscv_vor(__riscv_vor(vec_dst, __riscv_vsll(__riscv_vand(vec_srcG, ~7, vl), 2, vl), vl), __riscv_vsll(__riscv_vand(vec_srcR, ~7, vl), 7, vl), vl);
                vec_dst = __riscv_vor_mu(__riscv_vmsne(vec_srcA, 0, vl), vec_dst, vec_dst, 0x8000, vl);
            }
            __riscv_vse16(dst + i * dst_step + j, vec_dst, vl);
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int cvtBGRtoBGR5x5(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int scn, bool swapBlue, int greenBits)
{
    if ((scn != 3 && scn != 4) || (greenBits != 5 && greenBits != 6))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    return color::invoke(width, height, {cvtBGRtoBGR5x5_u}, reinterpret_cast<const uchar*>(src_data), src_step, reinterpret_cast<ushort*>(dst_data), dst_step, width, scn, swapBlue, greenBits);
}
} // cv::cv_hal_rvv::BGRtoBGR5x5

namespace BGR5x5toGray {
#undef cv_hal_cvtBGR5x5toGray
#define cv_hal_cvtBGR5x5toGray cv::cv_hal_rvv::BGR5x5toGray::cvtBGR5x5toGray

// the algorithm is copied from imgproc/src/color_rgb.simd.cpp,
// in the functor struct RGB5x52Gray
static inline int cvtBGR5x5toGray_u(int start, int end, const ushort * src, size_t src_step, uchar * dst, size_t dst_step, int width, int greenBits)
{
    static constexpr uint B2Y = 3735, G2Y = 19235, R2Y = 9798;

    src_step /= sizeof(ushort);

    for (int i = start; i < end; i++)
    {
        int vl;
        for (int j = 0; j < width; j += vl)
        {
            vl = __riscv_vsetvl_e16m2(width - j);
            auto vec_src = __riscv_vle16_v_u16m2(src + i * src_step + j, vl);

            auto vec_dstB = __riscv_vwcvtu_x(__riscv_vand(__riscv_vsll(vec_src, 3, vl), 0xF8, vl), vl);
            vuint32m4_t vec_dstG, vec_dstR;
            if (greenBits == 6)
            {
                vec_dstG = __riscv_vwcvtu_x(__riscv_vand(__riscv_vsrl(vec_src, 3, vl), 0xFC, vl), vl);
                vec_dstR = __riscv_vwcvtu_x(__riscv_vand(__riscv_vsrl(vec_src, 8, vl), 0xF8, vl), vl);
            }
            else
            {
                vec_dstG = __riscv_vwcvtu_x(__riscv_vand(__riscv_vsrl(vec_src, 2, vl), 0xF8, vl), vl);
                vec_dstR = __riscv_vwcvtu_x(__riscv_vand(__riscv_vsrl(vec_src, 7, vl), 0xF8, vl), vl);
            }

            auto vec_dst = __riscv_vncvt_x(__riscv_vnclipu(__riscv_vmadd(vec_dstB, B2Y, __riscv_vmadd(vec_dstG, G2Y, __riscv_vmul(vec_dstR, R2Y, vl), vl), vl), 15, __RISCV_VXRM_RNU, vl), vl);
            __riscv_vse8(dst + i * dst_step + j, vec_dst, vl);
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int cvtBGR5x5toGray(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int greenBits)
{
    if (greenBits != 5 && greenBits != 6)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    return color::invoke(width, height, {cvtBGR5x5toGray_u}, reinterpret_cast<const ushort*>(src_data), src_step, reinterpret_cast<uchar*>(dst_data), dst_step, width, greenBits);
}
} // cv::cv_hal_rvv::BGR5x5toGray

namespace GraytoBGR5x5 {
#undef cv_hal_cvtGraytoBGR5x5
#define cv_hal_cvtGraytoBGR5x5 cv::cv_hal_rvv::GraytoBGR5x5::cvtGraytoBGR5x5

// the algorithm is copied from imgproc/src/color_rgb.simd.cpp,
// in the functor struct Gray2RGB5x5
static inline int cvtGraytoBGR5x5_u(int start, int end, const uchar * src, size_t src_step, ushort * dst, size_t dst_step, int width, int greenBits)
{
    dst_step /= sizeof(ushort);

    for (int i = start; i < end; i++)
    {
        int vl;
        for (int j = 0; j < width; j += vl)
        {
            vl = __riscv_vsetvl_e8m2(width - j);
            auto vec_src = __riscv_vwcvtu_x(__riscv_vle8_v_u8m2(src + i * src_step + j, vl), vl);

            auto vec_dst = __riscv_vsrl(vec_src, 3, vl);
            if (greenBits == 6)
            {
                vec_dst = __riscv_vor(__riscv_vor(vec_dst, __riscv_vsll(__riscv_vand(vec_src, ~3, vl), 3, vl), vl), __riscv_vsll(vec_dst, 11, vl), vl);
            }
            else
            {
                vec_dst = __riscv_vor(__riscv_vor(vec_dst, __riscv_vsll(vec_dst, 5, vl), vl), __riscv_vsll(vec_dst, 10, vl), vl);
            }
            __riscv_vse16(dst + i * dst_step + j, vec_dst, vl);
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int cvtGraytoBGR5x5(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int greenBits)
{
    if (greenBits != 5 && greenBits != 6)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    return color::invoke(width, height, {cvtGraytoBGR5x5_u}, reinterpret_cast<const uchar*>(src_data), src_step, reinterpret_cast<ushort*>(dst_data), dst_step, width, greenBits);
}
} // cv::cv_hal_rvv::GraytoBGR5x5

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
    static inline void vsseg(uchar* a, int b, T c, T d, T e, T f, size_t g)
    {
        if (b == 3)
        {
            vuint8m1x3_t x{};
            x = __riscv_vset_v_u8m1_u8m1x3(x, 0, c);
            x = __riscv_vset_v_u8m1_u8m1x3(x, 1, d);
            x = __riscv_vset_v_u8m1_u8m1x3(x, 2, e);
            __riscv_vsseg3e8(a, x, g);
        }
        else
        {
            vuint8m1x4_t x{};
            x = __riscv_vset_v_u8m1_u8m1x4(x, 0, c);
            x = __riscv_vset_v_u8m1_u8m1x4(x, 1, d);
            x = __riscv_vset_v_u8m1_u8m1x4(x, 2, e);
            x = __riscv_vset_v_u8m1_u8m1x4(x, 3, f);
            __riscv_vsseg4e8(a, x, g);
        }
    }
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
    static inline void vsseg(ushort* a, int b, T c, T d, T e, T f, size_t g)
    {
        if (b == 3)
        {
            vuint16m2x3_t x{};
            x = __riscv_vset_v_u16m2_u16m2x3(x, 0, c);
            x = __riscv_vset_v_u16m2_u16m2x3(x, 1, d);
            x = __riscv_vset_v_u16m2_u16m2x3(x, 2, e);
            __riscv_vsseg3e16(a, x, g);
        }
        else
        {
            vuint16m2x4_t x{};
            x = __riscv_vset_v_u16m2_u16m2x4(x, 0, c);
            x = __riscv_vset_v_u16m2_u16m2x4(x, 1, d);
            x = __riscv_vset_v_u16m2_u16m2x4(x, 2, e);
            x = __riscv_vset_v_u16m2_u16m2x4(x, 3, f);
            __riscv_vsseg4e16(a, x, g);
        }
    }
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
    static inline void vsseg(float* a, int b, T c, T d, T e, T f, size_t g)
    {
        if (b == 3)
        {
            vfloat32m2x3_t x{};
            x = __riscv_vset_v_f32m2_f32m2x3(x, 0, c);
            x = __riscv_vset_v_f32m2_f32m2x3(x, 1, d);
            x = __riscv_vset_v_f32m2_f32m2x3(x, 2, e);
            __riscv_vsseg3e32(a, x, g);
        }
        else
        {
            vfloat32m2x4_t x{};
            x = __riscv_vset_v_f32m2_f32m2x4(x, 0, c);
            x = __riscv_vset_v_f32m2_f32m2x4(x, 1, d);
            x = __riscv_vset_v_f32m2_f32m2x4(x, 2, e);
            x = __riscv_vset_v_f32m2_f32m2x4(x, 3, f);
            __riscv_vsseg4e32(a, x, g);
        }
    }
    static inline T vcvt0(T a, size_t) { return a; }
    static inline T vcvt1(T a, T b, size_t, size_t d) { return __riscv_vfadd(a, b, d); }
    static inline T vsub(T a, float b, size_t c) { return __riscv_vfsub(a, b, c); }
    static inline T vmul(T a, float b, size_t c) { return __riscv_vfmul(a, b, c); }
    static inline T vmadd(T a, float b, T c, size_t d) { return __riscv_vfmadd(a, b, c, d); }
    static inline T vmv_v_x(float a, size_t b) { return __riscv_vfmv_v_f_f32m2(a, b); }
};

// the algorithm is copied from imgproc/src/color_yuv.simd.cpp,
// in the functor struct YCrCb2RGB_f and YCrCb2RGB_i
template<typename T>
static inline int cvtYUVtoBGR(int start, int end, const T * src, size_t src_step, T * dst, size_t dst_step, int width, int dcn, bool swapBlue, bool isCbCr)
{
    src_step /= sizeof(T);
    dst_step /= sizeof(T);

    decltype(rvv<T>::U2B) delta = typeid(T) == typeid(float) ? (T)1/2 : std::numeric_limits<T>::max() / 2 + 1;
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
        return color::invoke(width, height, {cvtYUVtoBGR<uchar>}, reinterpret_cast<const uchar*>(src_data), src_step, reinterpret_cast<uchar*>(dst_data), dst_step, width, dcn, swapBlue, isCbCr);
    case CV_16U:
        return color::invoke(width, height, {cvtYUVtoBGR<ushort>}, reinterpret_cast<const ushort*>(src_data), src_step, reinterpret_cast<ushort*>(dst_data), dst_step, width, dcn, swapBlue, isCbCr);
    case CV_32F:
        return color::invoke(width, height, {cvtYUVtoBGR<float>}, reinterpret_cast<const float*>(src_data), src_step, reinterpret_cast<float*>(dst_data), dst_step, width, dcn, swapBlue, isCbCr);
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
    static inline void vsseg(uchar* a, T b, T c, T d, size_t e)
    {
        vuint8m1x3_t x{};
        x = __riscv_vset_v_u8m1_u8m1x3(x, 0, b);
        x = __riscv_vset_v_u8m1_u8m1x3(x, 1, c);
        x = __riscv_vset_v_u8m1_u8m1x3(x, 2, d);
        __riscv_vsseg3e8(a, x, e);
    }
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
    static inline void vsseg(ushort* a, T b, T c, T d, size_t e)
    {
        vuint16m2x3_t x{};
        x = __riscv_vset_v_u16m2_u16m2x3(x, 0, b);
        x = __riscv_vset_v_u16m2_u16m2x3(x, 1, c);
        x = __riscv_vset_v_u16m2_u16m2x3(x, 2, d);
        __riscv_vsseg3e16(a, x, e);
    }
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
    static inline void vsseg(float* a, T b, T c, T d, size_t e)
    {
        vfloat32m2x3_t x{};
        x = __riscv_vset_v_f32m2_f32m2x3(x, 0, b);
        x = __riscv_vset_v_f32m2_f32m2x3(x, 1, c);
        x = __riscv_vset_v_f32m2_f32m2x3(x, 2, d);
        __riscv_vsseg3e32(a, x, e);
    }
    static inline T vcvt0(T a, size_t) { return a; }
    static inline T vcvt1(T a, size_t, size_t) { return a; }
    static inline T vssra(T a, size_t, size_t) { return a; }
    static inline T vsub(T a, T b, size_t c) { return __riscv_vfsub(a, b, c); }
    static inline T vmul(T a, float b, size_t c) { return __riscv_vfmul(a, b, c); }
    static inline T vmadd(T a, float b, T c, size_t d) { return __riscv_vfmadd(a, b, c, d); }
    static inline T vmv_v_x(float a, size_t b) { return __riscv_vfmv_v_f_f32m2(a, b); }
};

// the algorithm is copied from imgproc/src/color_yuv.simd.cpp,
// in the functor struct RGB2YCrCb_f and RGB2YCrCb_i
template<typename T>
static inline int cvtBGRtoYUV(int start, int end, const T * src, size_t src_step, T * dst, size_t dst_step, int width, int scn, bool swapBlue, bool isCbCr)
{
    src_step /= sizeof(T);
    dst_step /= sizeof(T);

    auto delta = rvv<T>::vmv_v_x(typeid(T) == typeid(float) ? (T)1/2 : (1 << 14) * (std::numeric_limits<T>::max() / 2 + 1), rvv<T>::vsetvlmax());
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
        return color::invoke(width, height, {cvtBGRtoYUV<uchar>}, reinterpret_cast<const uchar*>(src_data), src_step, reinterpret_cast<uchar*>(dst_data), dst_step, width, scn, swapBlue, isCbCr);
    case CV_16U:
        return color::invoke(width, height, {cvtBGRtoYUV<ushort>}, reinterpret_cast<const ushort*>(src_data), src_step, reinterpret_cast<ushort*>(dst_data), dst_step, width, scn, swapBlue, isCbCr);
    case CV_32F:
        return color::invoke(width, height, {cvtBGRtoYUV<float>}, reinterpret_cast<const float*>(src_data), src_step, reinterpret_cast<float*>(dst_data), dst_step, width, scn, swapBlue, isCbCr);
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
        throw;
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
    return color::invoke(dst_width, dst_height, {cvtSinglePlaneYUVtoBGR}, dst_data, dst_step, dst_width, src_step, src_data, dcn, swapBlue, uIdx, yIdx);
}

inline int cvtTwoPlaneYUVtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int dst_width, int dst_height, int dcn, bool swapBlue, int uIdx)
{
    if (dcn != 3 && dcn != 4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    const uchar* uv = src_data + src_step * static_cast<size_t>(dst_height);
    return color::invoke(dst_width, dst_height / 2, {cvtMultiPlaneYUVtoBGR}, dst_data, dst_step, dst_width, src_step, src_data, uv, uv, 0, 0, dcn, swapBlue, uIdx);
}

inline int cvtThreePlaneYUVtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int dst_width, int dst_height, int dcn, bool swapBlue, int uIdx)
{
    if (dcn != 3 && dcn != 4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    const uchar* u = src_data + src_step * static_cast<size_t>(dst_height);
    const uchar* v = src_data + src_step * static_cast<size_t>(dst_height + dst_height/4) + (dst_width/2) * ((dst_height % 4)/2);

    int ustepIdx = 0;
    int vstepIdx = dst_height % 4 == 2 ? 1 : 0;
    if (uIdx == 1) { std::swap(u ,v), std::swap(ustepIdx, vstepIdx); }

    return color::invoke(dst_width, dst_height / 2, {cvtMultiPlaneYUVtoBGR}, dst_data, dst_step, dst_width, src_step, src_data, u, v, ustepIdx, vstepIdx, dcn, swapBlue, -1);
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
        throw;
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

            if (evenRow)
            {
                vuint8m1_t uu, vv;
                bgrToUV42x(vl, b0, g0, r0, uu, vv);
                if (uIdx & 1)
                {
                    auto t = uu;
                    uu = vv, vv = t;
                }

                if (uIdx < 2)
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
    return color::invoke(width, height, {cvtBGRtoSinglePlaneYUV}, dst_data, dst_step, width, src_step, src_data, scn, swapBlue, uIdx, yIdx);
}

inline int cvtBGRtoTwoPlaneYUV(const uchar * src_data, size_t src_step,
                               uchar * y_data, size_t y_step, uchar * uv_data, size_t uv_step,
                               int width, int height,
                               int scn, bool swapBlue, int uIdx)
{
    if (y_step != uv_step || (scn != 3 && scn != 4))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    return color::invoke(width, height / 2, {cvtBGRtoMultiPlaneYUV}, y_data, uv_data, y_step, width, height, src_step, src_data, scn, swapBlue, uIdx == 2);
}

inline int cvtBGRtoThreePlaneYUV(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int scn, bool swapBlue, int uIdx)
{
    if (scn != 3 && scn != 4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    uchar* uv_data = dst_data + dst_step * static_cast<size_t>(height);
    return color::invoke(width, height / 2, {cvtBGRtoMultiPlaneYUV}, dst_data, uv_data, dst_step, width, height, src_step, src_data, scn, swapBlue, uIdx == 2 ? 3 : 2);
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
    asm volatile("fsrmi %0, 2 \n\t vsetvli zero,%2,e32,m2,ta,ma \n\t vfcvt.x.f.v %1,%3 \n\t fsrm %0"
                 : "=&r"(rd), "=vr"(sector)
                 : "r"(vl), "vr"(h));  // Rounding Mode: RDN

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
inline int cvtHSVtoBGR<float>(int start, int end, const float * src, size_t src_step, float * dst, size_t dst_step, int width, int dcn, bool swapBlue, bool /*isFullRange*/, bool isHSV)
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
        return color::invoke(width, height, {cvtHSVtoBGR<uchar>}, reinterpret_cast<const uchar*>(src_data), src_step, reinterpret_cast<uchar*>(dst_data), dst_step, width, dcn, swapBlue, isFullRange, isHSV);
    case CV_32F:
        return color::invoke(width, height, {cvtHSVtoBGR<float>}, reinterpret_cast<const float*>(src_data), src_step, reinterpret_cast<float*>(dst_data), dst_step, width, dcn, swapBlue, isFullRange, isHSV);
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
            vl = __riscv_vsetvl_e8m1(width - j);
            vint32m4_t b, g, r;
            if (scn == 3)
            {
                auto x = __riscv_vlseg3e8_v_u8m1x3(src + i * src_step + j * 3, vl);
                b = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4(__riscv_vget_v_u8m1x3_u8m1(x, 0), vl));
                g = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4(__riscv_vget_v_u8m1x3_u8m1(x, 1), vl));
                r = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4(__riscv_vget_v_u8m1x3_u8m1(x, 2), vl));
            }
            else
            {
                auto x = __riscv_vlseg4e8_v_u8m1x4(src + i * src_step + j * 4, vl);
                b = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4(__riscv_vget_v_u8m1x4_u8m1(x, 0), vl));
                g = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4(__riscv_vget_v_u8m1x4_u8m1(x, 1), vl));
                r = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4(__riscv_vget_v_u8m1x4_u8m1(x, 2), vl));
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

            vint32m4_t l, t;
            if (isHSV)
            {
                t = v;
            }
            else
            {
                l = __riscv_vdiv(__riscv_vadd(v, vmin, vl), 2, vl);
                t = __riscv_vmerge(__riscv_vrsub(__riscv_vadd(v, vmin, vl), std::numeric_limits<uchar>::max() * 2, vl), __riscv_vadd(v, vmin, vl), __riscv_vmslt(l, std::numeric_limits<uchar>::max() / 2 + 1, vl), vl);
            }
            auto s = __riscv_vssra(__riscv_vmul(diff, __riscv_vfcvt_x(__riscv_vfrdiv(__riscv_vfcvt_f(t, vl), 255 << 12, vl), vl), vl), 12, __RISCV_VXRM_RNU, vl);

            auto h = __riscv_vmadd(diff, 4, __riscv_vsub(r, g, vl), vl);
            h = __riscv_vmerge(h, __riscv_vmadd(diff, 2, __riscv_vsub(b, r, vl), vl), __riscv_vmseq(v, g, vl), vl);
            h = __riscv_vmerge(h, __riscv_vsub(g, b, vl), __riscv_vmseq(v, r, vl), vl);
            h = __riscv_vssra(__riscv_vmul(h, __riscv_vfcvt_x(__riscv_vfrdiv(__riscv_vfcvt_f(__riscv_vmul(diff, 6, vl), vl), isFullRange ? 256 << 12 : 180 << 12, vl), vl), vl), 12, __RISCV_VXRM_RNU, vl);
            h = __riscv_vadd_mu(__riscv_vmslt(h, 0, vl), h, h, isFullRange ? 256 : 180, vl);

            vuint8m1x3_t x{};
            x = __riscv_vset_v_u8m1_u8m1x3(x, 0, __riscv_vnclipu(__riscv_vnclipu(__riscv_vreinterpret_v_i32m4_u32m4(h), 0, __RISCV_VXRM_RNU, vl), 0, __RISCV_VXRM_RNU, vl));
            x = __riscv_vset_v_u8m1_u8m1x3(x, 1, __riscv_vncvt_x(__riscv_vncvt_x(__riscv_vreinterpret_v_i32m4_u32m4(isHSV ? s : l), vl), vl));
            x = __riscv_vset_v_u8m1_u8m1x3(x, 2, __riscv_vncvt_x(__riscv_vncvt_x(__riscv_vreinterpret_v_i32m4_u32m4(isHSV ? v : s), vl), vl));
            __riscv_vsseg3e8(dst + i * dst_step + j * 3, x, vl);
        }
    }

    return CV_HAL_ERROR_OK;
}

template<>
inline int cvtBGRtoHSV<float>(int start, int end, const float * src, size_t src_step, float * dst, size_t dst_step, int width, int scn, bool swapBlue, bool /*isFullRange*/, bool isHSV)
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

            vfloat32m2_t l, t;
            if (isHSV)
            {
                t = __riscv_vfadd(__riscv_vfabs(v, vl), FLT_EPSILON, vl);
            }
            else
            {
                l = __riscv_vfmul(__riscv_vfadd(v, vmin, vl), 0.5f, vl);
                t = __riscv_vmerge(__riscv_vfrsub(__riscv_vfadd(v, vmin, vl), 2.0f, vl), __riscv_vfadd(v, vmin, vl), __riscv_vmflt(l, 0.5f, vl), vl);
            }
            auto s = __riscv_vfdiv(diff, t, vl);
            diff = __riscv_vfrdiv(__riscv_vfadd(diff, FLT_EPSILON, vl), 60.0f, vl);

            auto h = __riscv_vfmadd(__riscv_vfsub(r, g, vl), diff, __riscv_vfmv_v_f_f32m2(240.0f, vl), vl);
            h = __riscv_vmerge(h, __riscv_vfmadd(__riscv_vfsub(b, r, vl), diff, __riscv_vfmv_v_f_f32m2(120.0f, vl), vl), __riscv_vmfeq(v, g, vl), vl);
            h = __riscv_vmerge(h, __riscv_vfmul(__riscv_vfsub(g, b, vl), diff, vl), __riscv_vmfeq(v, r, vl), vl);
            h = __riscv_vfadd_mu(__riscv_vmflt(h, 0, vl), h, h, 360.0f, vl);

            vfloat32m2x3_t x{};
            x = __riscv_vset_v_f32m2_f32m2x3(x, 0, h);
            x = __riscv_vset_v_f32m2_f32m2x3(x, 1, isHSV ? s : l);
            x = __riscv_vset_v_f32m2_f32m2x3(x, 2, isHSV ? v : s);
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
        return color::invoke(width, height, {cvtBGRtoHSV<uchar>}, reinterpret_cast<const uchar*>(src_data), src_step, reinterpret_cast<uchar*>(dst_data), dst_step, width, scn, swapBlue, isFullRange, isHSV);
    case CV_32F:
        return color::invoke(width, height, {cvtBGRtoHSV<float>}, reinterpret_cast<const float*>(src_data), src_step, reinterpret_cast<float*>(dst_data), dst_step, width, scn, swapBlue, isFullRange, isHSV);
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}
} // cv::cv_hal_rvv::BGRtoHSV

namespace XYZtoBGR {
#undef cv_hal_cvtXYZtoBGR
#define cv_hal_cvtXYZtoBGR cv::cv_hal_rvv::XYZtoBGR::cvtXYZtoBGR

template<typename T> struct rvv;
template<> struct rvv<uchar>
{
    using T = vuint8m1_t;
    static constexpr int XYZ2BGR_D65[] =
    {
          228,   -836,   4331,
        -3970,   7684,    170,
        13273,  -6296,  -2042
    };
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e8m1(); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e8m1(a); }
    static inline void vlseg(const uchar* a, T& b, T& c, T& d, size_t e){ auto x = __riscv_vlseg3e8_v_u8m1x3(a, e); b = __riscv_vget_v_u8m1x3_u8m1(x, 0), c = __riscv_vget_v_u8m1x3_u8m1(x, 1), d = __riscv_vget_v_u8m1x3_u8m1(x, 2); }
    static inline void vsseg(uchar* a, int b, T c, T d, T e, T f, size_t g)
    {
        if (b == 3)
        {
            vuint8m1x3_t x{};
            x = __riscv_vset_v_u8m1_u8m1x3(x, 0, c);
            x = __riscv_vset_v_u8m1_u8m1x3(x, 1, d);
            x = __riscv_vset_v_u8m1_u8m1x3(x, 2, e);
            __riscv_vsseg3e8(a, x, g);
        }
        else
        {
            vuint8m1x4_t x{};
            x = __riscv_vset_v_u8m1_u8m1x4(x, 0, c);
            x = __riscv_vset_v_u8m1_u8m1x4(x, 1, d);
            x = __riscv_vset_v_u8m1_u8m1x4(x, 2, e);
            x = __riscv_vset_v_u8m1_u8m1x4(x, 3, f);
            __riscv_vsseg4e8(a, x, g);
        }
    }
    static inline vint32m4_t vcvt0(T a, size_t b) { return __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf4(a, b)); }
    static inline T vcvt1(vint32m4_t a, size_t b, size_t c) { return __riscv_vnclipu(__riscv_vnclipu(__riscv_vreinterpret_v_i32m4_u32m4(__riscv_vmax(a, 0, c)), b, __RISCV_VXRM_RNU, c), 0, __RISCV_VXRM_RNU, c); }
    static inline vint32m4_t vmul(vint32m4_t a, int b, size_t c) { return __riscv_vmul(a, b, c); }
    static inline vint32m4_t vmadd(vint32m4_t a, int b, vint32m4_t c, size_t d) { return __riscv_vmadd(a, b, c, d); }
    static inline T vmv_v_x(uchar a, size_t b) { return __riscv_vmv_v_x_u8m1(a, b); }
};
template<> struct rvv<ushort>
{
    using T = vuint16m2_t;
    static constexpr int XYZ2BGR_D65[] =
    {
          228,   -836,   4331,
        -3970,   7684,    170,
        13273,  -6296,  -2042
    };
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e16m2(); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e16m2(a); }
    static inline void vlseg(const ushort* a, T& b, T& c, T& d, size_t e){ auto x = __riscv_vlseg3e16_v_u16m2x3(a, e); b = __riscv_vget_v_u16m2x3_u16m2(x, 0), c = __riscv_vget_v_u16m2x3_u16m2(x, 1), d = __riscv_vget_v_u16m2x3_u16m2(x, 2); }
    static inline void vsseg(ushort* a, int b, T c, T d, T e, T f, size_t g)
    {
        if (b == 3)
        {
            vuint16m2x3_t x{};
            x = __riscv_vset_v_u16m2_u16m2x3(x, 0, c);
            x = __riscv_vset_v_u16m2_u16m2x3(x, 1, d);
            x = __riscv_vset_v_u16m2_u16m2x3(x, 2, e);
            __riscv_vsseg3e16(a, x, g);
        }
        else
        {
            vuint16m2x4_t x{};
            x = __riscv_vset_v_u16m2_u16m2x4(x, 0, c);
            x = __riscv_vset_v_u16m2_u16m2x4(x, 1, d);
            x = __riscv_vset_v_u16m2_u16m2x4(x, 2, e);
            x = __riscv_vset_v_u16m2_u16m2x4(x, 3, f);
            __riscv_vsseg4e16(a, x, g);
        }
    }
    static inline vint32m4_t vcvt0(T a, size_t b) { return __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vzext_vf2(a, b)); }
    static inline T vcvt1(vint32m4_t a, size_t b, size_t c) { return __riscv_vnclipu(__riscv_vreinterpret_v_i32m4_u32m4(__riscv_vmax(a, 0, c)), b, __RISCV_VXRM_RNU, c); }
    static inline vint32m4_t vmul(vint32m4_t a, int b, size_t c) { return __riscv_vmul(a, b, c); }
    static inline vint32m4_t vmadd(vint32m4_t a, int b, vint32m4_t c, size_t d) { return __riscv_vmadd(a, b, c, d); }
    static inline T vmv_v_x(ushort a, size_t b) { return __riscv_vmv_v_x_u16m2(a, b); }
};
template<> struct rvv<float>
{
    using T = vfloat32m2_t;
    static constexpr float XYZ2BGR_D65[] =
    {
         0.055648f, -0.204043f,  1.057311f,
        -0.969256f,  1.875991f,  0.041556f,
         3.240479f, -1.53715f , -0.498535f
    };
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e32m2(); }
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e32m2(a); }
    static inline void vlseg(const float* a, T& b, T& c, T& d, size_t e){ auto x = __riscv_vlseg3e32_v_f32m2x3(a, e); b = __riscv_vget_v_f32m2x3_f32m2(x, 0), c = __riscv_vget_v_f32m2x3_f32m2(x, 1), d = __riscv_vget_v_f32m2x3_f32m2(x, 2); }
    static inline void vsseg(float* a, int b, T c, T d, T e, T f, size_t g)
    {
        if (b == 3)
        {
            vfloat32m2x3_t x{};
            x = __riscv_vset_v_f32m2_f32m2x3(x, 0, c);
            x = __riscv_vset_v_f32m2_f32m2x3(x, 1, d);
            x = __riscv_vset_v_f32m2_f32m2x3(x, 2, e);
            __riscv_vsseg3e32(a, x, g);
        }
        else
        {
            vfloat32m2x4_t x{};
            x = __riscv_vset_v_f32m2_f32m2x4(x, 0, c);
            x = __riscv_vset_v_f32m2_f32m2x4(x, 1, d);
            x = __riscv_vset_v_f32m2_f32m2x4(x, 2, e);
            x = __riscv_vset_v_f32m2_f32m2x4(x, 3, f);
            __riscv_vsseg4e32(a, x, g);
        }
    }
    static inline T vcvt0(T a, size_t) { return a; }
    static inline T vcvt1(T a, size_t, size_t) { return a; }
    static inline T vmul(T a, float b, size_t c) { return __riscv_vfmul(a, b, c); }
    static inline T vmadd(T a, float b, T c, size_t d) { return __riscv_vfmadd(a, b, c, d); }
    static inline T vmv_v_x(float a, size_t b) { return __riscv_vfmv_v_f_f32m2(a, b); }
};

// the algorithm is copied from imgproc/src/color_lab.cpp,
// in the functor struct XYZ2RGB_f and XYZ2RGB_i
template<typename T>
static inline int cvtXYZtoBGR(int start, int end, const T * src, size_t src_step, T * dst, size_t dst_step, int width, int dcn, bool swapBlue)
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
            typename rvv<T>::T vec_srcX_T, vec_srcY_T, vec_srcZ_T;
            rvv<T>::vlseg(src + i * src_step + j * 3, vec_srcX_T, vec_srcY_T, vec_srcZ_T, vl);
            auto vec_srcX = rvv<T>::vcvt0(vec_srcX_T, vl);
            auto vec_srcY = rvv<T>::vcvt0(vec_srcY_T, vl);
            auto vec_srcZ = rvv<T>::vcvt0(vec_srcZ_T, vl);

            auto vec_dstB = rvv<T>::vmadd(vec_srcX, rvv<T>::XYZ2BGR_D65[0], rvv<T>::vmadd(vec_srcY, rvv<T>::XYZ2BGR_D65[1], rvv<T>::vmul(vec_srcZ, rvv<T>::XYZ2BGR_D65[2], vl), vl), vl);
            auto vec_dstG = rvv<T>::vmadd(vec_srcX, rvv<T>::XYZ2BGR_D65[3], rvv<T>::vmadd(vec_srcY, rvv<T>::XYZ2BGR_D65[4], rvv<T>::vmul(vec_srcZ, rvv<T>::XYZ2BGR_D65[5], vl), vl), vl);
            auto vec_dstR = rvv<T>::vmadd(vec_srcX, rvv<T>::XYZ2BGR_D65[6], rvv<T>::vmadd(vec_srcY, rvv<T>::XYZ2BGR_D65[7], rvv<T>::vmul(vec_srcZ, rvv<T>::XYZ2BGR_D65[8], vl), vl), vl);
            if (swapBlue)
            {
                auto t = vec_dstB;
                vec_dstB = vec_dstR, vec_dstR = t;
            }
            rvv<T>::vsseg(dst + i * dst_step + j * dcn, dcn, rvv<T>::vcvt1(vec_dstB, 12, vl), rvv<T>::vcvt1(vec_dstG, 12, vl), rvv<T>::vcvt1(vec_dstR, 12, vl), alpha, vl);
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int cvtXYZtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int depth, int dcn, bool swapBlue)
{
    if (dcn != 3 && dcn != 4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    switch (depth)
    {
    case CV_8U:
        return color::invoke(width, height, {cvtXYZtoBGR<uchar>}, reinterpret_cast<const uchar*>(src_data), src_step, reinterpret_cast<uchar*>(dst_data), dst_step, width, dcn, swapBlue);
    case CV_16U:
        return color::invoke(width, height, {cvtXYZtoBGR<ushort>}, reinterpret_cast<const ushort*>(src_data), src_step, reinterpret_cast<ushort*>(dst_data), dst_step, width, dcn, swapBlue);
    case CV_32F:
        return color::invoke(width, height, {cvtXYZtoBGR<float>}, reinterpret_cast<const float*>(src_data), src_step, reinterpret_cast<float*>(dst_data), dst_step, width, dcn, swapBlue);
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}
} // cv::cv_hal_rvv::XYZtoBGR

namespace BGRtoXYZ {
#undef cv_hal_cvtBGRtoXYZ
#define cv_hal_cvtBGRtoXYZ cv::cv_hal_rvv::BGRtoXYZ::cvtBGRtoXYZ

template<typename T> struct rvv;
template<> struct rvv<uchar>
{
    using T = vuint8m1_t;
    static constexpr uint BGR2XYZ_D65[] =
    {
         739,  1465,  1689,
         296,  2929,   871,
        3892,   488,    79
    };
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
    static inline void vsseg(uchar* a, T b, T c, T d, size_t e)
    {
        vuint8m1x3_t x{};
        x = __riscv_vset_v_u8m1_u8m1x3(x, 0, b);
        x = __riscv_vset_v_u8m1_u8m1x3(x, 1, c);
        x = __riscv_vset_v_u8m1_u8m1x3(x, 2, d);
        __riscv_vsseg3e8(a, x, e);
    }
    static inline vuint32m4_t vcvt0(T a, size_t b) { return __riscv_vzext_vf4(a, b); }
    static inline T vcvt1(vuint32m4_t a, size_t b, size_t c) { return __riscv_vnclipu(__riscv_vnclipu(a, b, __RISCV_VXRM_RNU, c), 0, __RISCV_VXRM_RNU, c); }
    static inline vuint32m4_t vmul(vuint32m4_t a, uint b, size_t c) { return __riscv_vmul(a, b, c); }
    static inline vuint32m4_t vmadd(vuint32m4_t a, uint b, vuint32m4_t c, size_t d) { return __riscv_vmadd(a, b, c, d); }
};
template<> struct rvv<ushort>
{
    using T = vuint16m2_t;
    static constexpr uint BGR2XYZ_D65[] =
    {
         739,  1465,  1689,
         296,  2929,   871,
        3892,   488,    79
    };
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
    static inline void vsseg(ushort* a, T b, T c, T d, size_t e)
    {
        vuint16m2x3_t x{};
        x = __riscv_vset_v_u16m2_u16m2x3(x, 0, b);
        x = __riscv_vset_v_u16m2_u16m2x3(x, 1, c);
        x = __riscv_vset_v_u16m2_u16m2x3(x, 2, d);
        __riscv_vsseg3e16(a, x, e);
    }
    static inline vuint32m4_t vcvt0(T a, size_t b) { return __riscv_vzext_vf2(a, b); }
    static inline T vcvt1(vuint32m4_t a, size_t b, size_t c) { return __riscv_vnclipu(a, b, __RISCV_VXRM_RNU, c); }
    static inline vuint32m4_t vmul(vuint32m4_t a, uint b, size_t c) { return __riscv_vmul(a, b, c); }
    static inline vuint32m4_t vmadd(vuint32m4_t a, uint b, vuint32m4_t c, size_t d) { return __riscv_vmadd(a, b, c, d); }
};
template<> struct rvv<float>
{
    using T = vfloat32m2_t;
    static constexpr float BGR2XYZ_D65[] =
    {
        0.180423f, 0.357580f, 0.412453f,
        0.072169f, 0.715160f, 0.212671f,
        0.950227f, 0.119193f, 0.019334f
    };
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
    static inline void vsseg(float* a, T b, T c, T d, size_t e)
    {
        vfloat32m2x3_t x{};
        x = __riscv_vset_v_f32m2_f32m2x3(x, 0, b);
        x = __riscv_vset_v_f32m2_f32m2x3(x, 1, c);
        x = __riscv_vset_v_f32m2_f32m2x3(x, 2, d);
        __riscv_vsseg3e32(a, x, e);
    }
    static inline T vcvt0(T a, size_t) { return a; }
    static inline T vcvt1(T a, size_t, size_t) { return a; }
    static inline T vmul(T a, float b, size_t c) { return __riscv_vfmul(a, b, c); }
    static inline T vmadd(T a, float b, T c, size_t d) { return __riscv_vfmadd(a, b, c, d); }
};

// the algorithm is copied from imgproc/src/color_lab.cpp,
// in the functor struct RGB2XYZ_f and RGB2XYZ_i
template<typename T>
static inline int cvtBGRtoXYZ(int start, int end, const T * src, size_t src_step, T * dst, size_t dst_step, int width, int scn, bool swapBlue)
{
    src_step /= sizeof(T);
    dst_step /= sizeof(T);

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

            auto vec_dstX = rvv<T>::vmadd(vec_srcB, rvv<T>::BGR2XYZ_D65[0], rvv<T>::vmadd(vec_srcG, rvv<T>::BGR2XYZ_D65[1], rvv<T>::vmul(vec_srcR, rvv<T>::BGR2XYZ_D65[2], vl), vl), vl);
            auto vec_dstY = rvv<T>::vmadd(vec_srcB, rvv<T>::BGR2XYZ_D65[3], rvv<T>::vmadd(vec_srcG, rvv<T>::BGR2XYZ_D65[4], rvv<T>::vmul(vec_srcR, rvv<T>::BGR2XYZ_D65[5], vl), vl), vl);
            auto vec_dstZ = rvv<T>::vmadd(vec_srcB, rvv<T>::BGR2XYZ_D65[6], rvv<T>::vmadd(vec_srcG, rvv<T>::BGR2XYZ_D65[7], rvv<T>::vmul(vec_srcR, rvv<T>::BGR2XYZ_D65[8], vl), vl), vl);
            rvv<T>::vsseg(dst + i * dst_step + j * 3, rvv<T>::vcvt1(vec_dstX, 12, vl), rvv<T>::vcvt1(vec_dstY, 12, vl), rvv<T>::vcvt1(vec_dstZ, 12, vl), vl);
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int cvtBGRtoXYZ(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int depth, int scn, bool swapBlue)
{
    if (scn != 3 && scn != 4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    switch (depth)
    {
    case CV_8U:
        return color::invoke(width, height, {cvtBGRtoXYZ<uchar>}, reinterpret_cast<const uchar*>(src_data), src_step, reinterpret_cast<uchar*>(dst_data), dst_step, width, scn, swapBlue);
    case CV_16U:
        return color::invoke(width, height, {cvtBGRtoXYZ<ushort>}, reinterpret_cast<const ushort*>(src_data), src_step, reinterpret_cast<ushort*>(dst_data), dst_step, width, scn, swapBlue);
    case CV_32F:
        return color::invoke(width, height, {cvtBGRtoXYZ<float>}, reinterpret_cast<const float*>(src_data), src_step, reinterpret_cast<float*>(dst_data), dst_step, width, scn, swapBlue);
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}
} // cv::cv_hal_rvv::BGRtoXYZ

namespace LabTable
{
    class Tab
    {
    private:
        // the algorithm is copied from imgproc/src/color_lab.cpp,
        // in the function static bool createLabTabs
        Tab()
        {
            float f[GAMMA_TAB_SIZE + 1], g[GAMMA_TAB_SIZE + 1], ig[GAMMA_TAB_SIZE + 1];
            for (int i = 0; i <= GAMMA_TAB_SIZE; i++)
            {
                float x = i * 1.0f / GAMMA_TAB_SIZE;
                f[i] = applyCbrt(x);
                g[i] = applyGamma(x);
                ig[i] = applyInvGamma(x);
            }
            LabCbrtTab = splineBuild(f, GAMMA_TAB_SIZE);
            sRGBGammaTab = splineBuild(g, GAMMA_TAB_SIZE);
            sRGBInvGammaTab = splineBuild(ig, GAMMA_TAB_SIZE);

            for (int i = 0; i < 3072; i++)
            {
                float x = i * 1.0f / (255*8);
                LabCbrtTab_b[i] = (ushort)roundeven((1 << 15) * applyCbrt(x));
            }
            // tweak to imitate the error of cv::softfloat, or bitExactness tests won't pass
            LabCbrtTab_b[324] -= 1, LabCbrtTab_b[2079] -= 1;

            for (int i = 0; i < 256; i++)
            {
                float x = i / 255.0f;
                sRGBGammaTab_b[i] = (ushort)roundeven(2040 * applyGamma(x));
            }
            for (int i = 0; i < INV_GAMMA_TAB_SIZE; i++)
            {
                float x = i * 1.0f / INV_GAMMA_TAB_SIZE;
                sRGBInvGammaTab_b[i] = (ushort)roundeven(255 * applyInvGamma(x));
            }

            for (int i = 0; i < 256; i++)
            {
                float li = i * 100.0f / 255.0f, yy, fy;
                if ( i <= 20)
                {
                    yy = li / 903.3f;
                    fy = 7.787f * yy + 16.0f / 116.0f;
                }
                else
                {
                    fy = (li + 16.0f) / 116.0f;
                    yy = fy * fy * fy;
                }
                LabToYF_b[i*2  ] = (short)roundeven(yy * LAB_BASE);
                LabToYF_b[i*2+1] = (short)roundeven(fy * LAB_BASE);
            }

            for (int LL = 0; LL < 256; LL++)
            {
                float L = LL * 100.0f / 255.0f;
                for (int uu = 0; uu < 256; uu++)
                {
                    float u = uu*354.0f/255 - 134;
                    float up = 9.0f*(u + L*2.5719122887f);
                    LuToUp_b[LL*256+uu] = (int)roundeven(up*float(LAB_BASE/1024));
                }
                for (int vv = 0; vv < 256; vv++)
                {
                    float v = vv*262.0f/255 - 140;
                    float vp = 0.25f/(v + L*6.0884485245f);
                    if (vp >  0.25f) vp =  0.25f;
                    if (vp < -0.25f) vp = -0.25f;
                    LvToVp_b[LL*256+vv] = (int)roundeven(vp*float(LAB_BASE*1024));
                }
            }
            // tweak
            #ifdef __clang__
            static constexpr int error0[] = {17985,1,20935,1,32356,-1,35522,-1,36804,1,37916,1,39152,-1,39885,1,40951,-1,40997,1,41167,1,42063,1,43777,-1,45059,-1,45229,-1,46642,-1,47793,1,48140,-1,49422,-1,51090,1,52287,1,54256,1,58017,1,58102,1,58534,-1,58619,-1,59901,-1,60712,1,63446,-1,64179,1,65075,-1,65160,-1,65376,-1,65461,-1};
            static constexpr int error1[] = {394,1,1401,1,1406,1,1426,1,1863,1,1911,1,1913,1,2164,1,2167,1,2198,1,2427,1,2671,1,2672,1,2674,1,2925,1,2928,1,2929,1,3430,1,3432,1,3433,3,3436,1,3437,1,3690,1,3778,1,3927,-1,3940,-2,3943,-1,4193,1,4194,2,4199,1,4433,-1,4443,-1,4444,-1,4447,-1,4450,-2,4451,-1,4452,-1,4469,-1,4532,-1,4616,1,4696,1,4699,1,4701,2,4704,1,4709,1,4952,-1,4955,-3,4958,-1,4959,-1,5211,2,5212,1,5460,-1,5461,-1,5462,-3,5465,-3,5466,-1,5467,-1,5483,-1,5714,1,5716,3,5719,1,5721,1,5726,1,5969,-1,5972,-1,5973,-1,5975,-1,6471,-1,6476,-2,6477,-3,6480,-2,6481,-1,6482,-1,6490,-1,6662,1,6725,1,6728,1,6730,1,6733,3,6735,1,6738,1,6742,1,6749,1,6981,1,6982,1,6983,2,6984,6,6987,6,6988,2,6989,1,6990,1,6991,1,6992,1,7001,1,7151,1,7237,1,7238,1,7241,1,7491,-1,7494,-3,7729,-1,7738,-1,7743,-1,7744,-2,7745,-4,7748,-4,7749,-1,7754,-1,7991,1,7996,1,7997,1,7998,2,7999,6,8002,2,8003,1,8005,1,8045,1,8251,1,8252,1,8255,3,8256,1,8261,1,8506,-1,8510,-1,8744,-1,8755,-1,8758,-1,8759,-1,8760,-5,8763,-3,8764,-1,8765,-1,8766,-1,8768,-1,8773,-1,8778,-1,8986,1,9000,1,9010,1,9011,1,9012,1,9013,3,9016,9,9017,2,9018,1,9019,1,9025,1,9262,1,9264,1,9265,1,9266,1,9267,2,9270,2,9271,1,9272,1,9772,-1,9773,-1,9774,-2,9777,-6,9778,-2,9784,-1,9801,-1,9978,-1,10026,1,10027,2,10028,4,10031,6,10032,2,10033,2,10040,1,10273,1,10275,1,10278,1,10281,1,10282,3,10285,2,10286,1,10288,1,10289,1,10295,1,10533,-1,10534,-1,10535,-2,10536,-8,10538,-7,10539,-1,10540,-1,10541,-1,10542,-1,10545,-1,10549,-1,10705,-1,10778,-1,10783,-1,10785,-1,10786,-1,10787,-1,10788,-2,10789,-5,10792,-7,10793,-2,10794,-1,10795,-1,10805,-1,11020,1,11031,1,11034,1,11036,1,11038,1,11039,1,11041,1,11042,2,11043,7,11046,5,11047,2,11048,1,11049,1,11054,1,11061,1,11296,2,11297,6,11299,7,11300,2,11301,1,11313,1,11548,1,11553,1,11554,1,11802,-1,11803,-1,11804,-2,11807,-1,11810,-1,11811,-1,11814,-1,12034,-1,12038,-1,12040,-1,12056,-1,12057,-2,12058,-9,12061,-2,12062,-1,12067,-1,12072,-1,12310,1,12564,-1,12565,-2,12568,-2,12569,-1,12574,-1,12577,-1,12813,-1,12814,-1,12816,-1,12818,-2,12819,-8,12822,-2,12823,-1,13063,1,13067,1,13069,1,13070,1,13071,2,13072,5,13075,10,13076,3,13077,1,13080,1,13083,1,13086,1,13319,1,13323,1,13324,1,13325,2,13326,5,13329,4,13330,2,13332,1,13578,1,13579,1,13580,4,13583,1,13612,1,13826,1,13828,1,13829,1,13830,1,13831,2,13832,4,13833,9,13836,20,13837,6,13838,3,13839,1,13840,1,13841,1,13842,1,13845,1,13846,1,13848,1,13850,1,14081,-1,14083,-1,14084,-1,14085,-2,14086,-4,14087,-10,14090,-9,14091,-3,14092,-2,14093,-1,14094,-1,14096,-1,14097,-1,14336,1,14340,1,14341,3,14344,1,14345,1,14351,1,14592,2,14593,4,14594,9,14597,19,14598,6,14599,2,14600,1,14601,1,14602,1,14608,1,14609,1,14613,1,14614,1,14616,1,14848,-3,14851,-3,14852,-1,14853,-1,14860,-1,15105,5,15106,2,15107,1,15108,1,15109,1,15112,1,15360,-1,15361,-1,15365,-1,15616,1,15872,1,15876,1,15879,1,15887,1,15889,1,16135,-1,16218,-1,16643,-1,16658,-1,16714,-1,16730,-1,17929,1,18434,1,18708,-1,19457,-1,21257,1,21267,1,26382,-1,28276,-1,28960,1,30006,-1,30244,1,32854,1,33551,-1,34578,1,39762,1,40300,-1,42328,1,56165,-1,56403,-1,58626,-1,60298,1,63317,1,64183,-1,65410,1};
            for (size_t i = 0; i < sizeof(error0) / sizeof(int); i += 2)
                LuToUp_b[error0[i]] += error0[i + 1];
            for (size_t i = 0; i < sizeof(error1) / sizeof(int); i += 2)
                LvToVp_b[error1[i]] += error1[i + 1];
            #endif

            static constexpr float BGR2XYZ[] =
            {
                0.180423f / 0.950456f, 0.357580f / 0.950456f, 0.412453f / 0.950456f,
                0.072169f            , 0.715160f            , 0.212671f            ,
                0.950227f / 1.088754f, 0.119193f / 1.088754f, 0.019334f / 1.088754f
            };
            static constexpr float BGR2XYZ_D65[] =
            {
                0.180423f, 0.357580f, 0.412453f,
                0.072169f, 0.715160f, 0.212671f,
                0.950227f, 0.119193f, 0.019334f
            };
            short RGB2Labprev[LAB_LUT_DIM*LAB_LUT_DIM*LAB_LUT_DIM*3], RGB2Luvprev[LAB_LUT_DIM*LAB_LUT_DIM*LAB_LUT_DIM*3];
            for (int p = 0; p < LAB_LUT_DIM; p++)
            {
                for (int q = 0; q < LAB_LUT_DIM; q++)
                {
                    for (int r = 0; r < LAB_LUT_DIM; r++)
                    {
                        int idx = p*3 + q*LAB_LUT_DIM*3 + r*LAB_LUT_DIM*LAB_LUT_DIM*3;
                        float R = applyGamma(p / 32.0f);
                        float G = applyGamma(q / 32.0f);
                        float B = applyGamma(r / 32.0f);

                        {
                            float X = R*BGR2XYZ[0] + G*BGR2XYZ[1] + B*BGR2XYZ[2];
                            float Y = R*BGR2XYZ[3] + G*BGR2XYZ[4] + B*BGR2XYZ[5];
                            float Z = R*BGR2XYZ[6] + G*BGR2XYZ[7] + B*BGR2XYZ[8];

                            float FX = applyCbrt(X);
                            float FY = applyCbrt(Y);
                            float FZ = applyCbrt(Z);

                            float L = Y > 0.008856f ? (116.0f*FY - 16.0f) : (903.3f*Y);
                            float a = 500.0f * (FX - FY);
                            float b = 200.0f * (FY - FZ);

                            RGB2Labprev[idx]   = (short)(roundeven(LAB_BASE*L/100.0f));
                            RGB2Labprev[idx+1] = (short)(roundeven(LAB_BASE*(a+128.0f)/256.0f));
                            RGB2Labprev[idx+2] = (short)(roundeven(LAB_BASE*(b+128.0f)/256.0f));
                        }
                        {
                            float X = R*BGR2XYZ_D65[0] + G*BGR2XYZ_D65[1] + B*BGR2XYZ_D65[2];
                            float Y = R*BGR2XYZ_D65[3] + G*BGR2XYZ_D65[4] + B*BGR2XYZ_D65[5];
                            float Z = R*BGR2XYZ_D65[6] + G*BGR2XYZ_D65[7] + B*BGR2XYZ_D65[8];

                            float L = applyCbrt(Y);
                            L = L*116.0f - 16.0f;

                            float d = 52.0f/std::max(X + 15.0f * Y + 3.0f * Z, FLT_EPSILON);
                            float u = L*(X*d - 2.5719122887f);
                            float v = L*(2.25f*Y*d - 6.0884485245f);

                            RGB2Luvprev[idx  ] = (short)roundeven(LAB_BASE*L/100.0f);
                            RGB2Luvprev[idx+1] = (short)roundeven(LAB_BASE*(u+134.0f)/354.0f);
                            RGB2Luvprev[idx+2] = (short)roundeven(LAB_BASE*(v+140.0f)/262.0f);
                        }
                    }
                }
            }
            // tweak
            static constexpr int error2[] = {37,-1,124,-1,503,-1,4150,1,5548,1,6544,1,6659,1,8625,-1,11704,1,16108,-1,16347,-1,16446,-1,18148,1,19624,-1,22731,-1,23479,1,24001,1,24291,-1,25199,-1,25352,-1,27402,-1,28485,-1,29788,1,29807,-1,32149,-1,33451,-1,33974,-1,38265,-1,38403,-1,41038,-1,41279,1,41824,-1,42856,-1,48096,-1,49201,-1,50562,-1,51054,-1,51550,-1,51821,1,56973,-1,57283,1,62335,-1,67867,-1,70692,-1,71194,-1,71662,1,71815,1,72316,-1,73487,1,75722,-1,75959,1,82290,-1,82868,-1,83208,-1,83534,-1,84217,-1,85793,1,86683,-1,87939,-1,89143,1,90258,-1,91432,-1,92302,1,92325,-1,92572,1,93143,-1,93731,-1,94142,-1,95244,-1,96025,-1,96950,-1,97758,-1,102409,-1,104165,-1};
            static constexpr int error3[] = {32,-1,5246,-1,6662,-1,7837,1,8625,-1,11969,1,15290,1,19142,1,19588,1,21707,-1,22731,-1,24291,-1,25922,-1,27402,-1,28485,-1,29878,-1,32405,-1,36227,1,38265,-1,38296,1,38403,-1,41795,1,41867,1,43796,1,48096,-1,50562,-1,51054,-1,54496,1,55328,-1,56973,-1,58594,1,61568,1,66512,-1,68543,-1,68615,1,70105,-1,70692,-1,74924,1,76336,-1,78781,1,79259,-1,80855,1,81662,1,82290,-1,83208,-1,84370,1,86293,1,87263,-1,87939,-1,89942,-1,90258,-1,92101,-1,92325,-1,95244,-1,97556,1,97758,-1,97769,1,98455,1,104087,-1,106997,-1};
            for (size_t i = 0; i < sizeof(error2) / sizeof(int); i += 2)
                RGB2Labprev[error2[i]] += error2[i + 1];
            for (size_t i = 0; i < sizeof(error3) / sizeof(int); i += 2)
                RGB2Luvprev[error3[i]] += error3[i + 1];
            #ifdef __clang__
            RGB2Labprev[9704] -= 1, RGB2Labprev[41279] -= 1, RGB2Labprev[71194] += 1, RGB2Labprev[73487] -= 1, RGB2Labprev[85793] -= 1;
            RGB2Luvprev[36227] -= 1, RGB2Luvprev[38587] += 1;
            #endif
            for (int p = 0; p < LAB_LUT_DIM; p++)
                for (int q = 0; q < LAB_LUT_DIM; q++)
                    for (int r = 0; r < LAB_LUT_DIM; r++)
                        for (int p_ = 0; p_ < 2; ++p_)
                            for (int q_ = 0; q_ < 2; ++q_)
                                for (int r_ = 0; r_ < 2; ++r_)
                                {
                                    int idxold = std::min(p+p_, (int)(LAB_LUT_DIM-1))*3;
                                    idxold += std::min(q+q_, (int)(LAB_LUT_DIM-1))*LAB_LUT_DIM*3;
                                    idxold += std::min(r+r_, (int)(LAB_LUT_DIM-1))*LAB_LUT_DIM*LAB_LUT_DIM*3;
                                    int idxnew = p*3*8 + q*LAB_LUT_DIM*3*8 + r*LAB_LUT_DIM*LAB_LUT_DIM*3*8+4*p_+2*q_+r_;
                                    RGB2LabLUT[idxnew]    = RGB2Labprev[idxold];
                                    RGB2LabLUT[idxnew+8]  = RGB2Labprev[idxold+1];
                                    RGB2LabLUT[idxnew+16] = RGB2Labprev[idxold+2];
                                    RGB2LuvLUT[idxnew]    = RGB2Luvprev[idxold];
                                    RGB2LuvLUT[idxnew+8]  = RGB2Luvprev[idxold+1];
                                    RGB2LuvLUT[idxnew+16] = RGB2Luvprev[idxold+2];
                                }

            for (int p = 0; p < TRILINEAR_BASE; p++)
            {
                int pp = TRILINEAR_BASE - p;
                for (int q = 0; q < TRILINEAR_BASE; q++)
                {
                    int qq = TRILINEAR_BASE - q;
                    for (int r = 0; r < TRILINEAR_BASE; r++)
                    {
                        short rr = TRILINEAR_BASE - r;
                        short* w = &trilinearLUT[8*p + 8*TRILINEAR_BASE*q + 8*TRILINEAR_BASE*TRILINEAR_BASE*r];
                        w[0]  = pp * qq * rr; w[1]  = pp * qq * r ; w[2]  = pp * q  * rr; w[3]  = pp * q  * r ;
                        w[4]  = p  * qq * rr; w[5]  = p  * qq * r ; w[6]  = p  * q  * rr; w[7]  = p  * q  * r ;
                    }
                }
            }
        }

        ~Tab()
        {
            delete[] LabCbrtTab;
            delete[] sRGBGammaTab;
            delete[] sRGBInvGammaTab;
        }

        const float * splineBuild(const float* f, int n)
        {
            float* tab = new float[n * 4];
            tab[0] = tab[1] = 0.0f;
            for (int i = 1; i < n; i++)
            {
                float t = (f[i+1] - f[i]*2 + f[i-1])*3;
                float l = 1/(4 - tab[(i-1)*4]);
                tab[i*4] = l; tab[i*4+1] = (t - tab[(i-1)*4+1])*l;
            }

            float cn = 0;
            for (int j = 0; j < n; j++)
            {
                int i = n - j - 1;
                float c = tab[i*4+1] - tab[i*4]*cn;
                float b = f[i+1] - f[i] - (cn + c*2)/3;
                float d = (cn - c)/3;
                tab[i*4] = f[i]; tab[i*4+1] = b;
                tab[i*4+2] = c; tab[i*4+3] = d;
                cn = c;
            }
            return tab;
        }

        inline float applyCbrt(float x)
        {
            return x < 0.008856f ? x * 7.787f + (16.0f/116.0f) : std::cbrt(x);
        }

        inline float applyGamma(float x)
        {
            return x <= 0.04045f ? x*(1.f/12.92f) : (float)std::pow((double)(x + 0.055)*(1./1.055), 2.4);
        }

        inline float applyInvGamma(float x)
        {
            return x <= 0.0031308 ? x*12.92f : (float)(1.055*std::pow((double)x, 1./2.4) - 0.055);
        }

    public:
        static constexpr int GAMMA_TAB_SIZE = 1024, INV_GAMMA_TAB_SIZE = 4096;
        static constexpr int LAB_BASE = 1 << 14, TRILINEAR_BASE = 16, LAB_LUT_DIM = 33;

        const float* LabCbrtTab, *sRGBGammaTab, *sRGBInvGammaTab;
        ushort LabCbrtTab_b[3072];
        ushort sRGBGammaTab_b[256], sRGBInvGammaTab_b[INV_GAMMA_TAB_SIZE];
        short LabToYF_b[256*2];
        int LuToUp_b[256*256], LvToVp_b[256*256];
        short RGB2LabLUT[LAB_LUT_DIM*LAB_LUT_DIM*LAB_LUT_DIM*3*8], RGB2LuvLUT[LAB_LUT_DIM*LAB_LUT_DIM*LAB_LUT_DIM*3*8];
        short trilinearLUT[TRILINEAR_BASE*TRILINEAR_BASE*TRILINEAR_BASE*8];

        static Tab& instance()
        {
            static Tab tab;
            return tab;
        }

        static vfloat32m2_t splineInterpolate(int vl, vfloat32m2_t x, const float* tab, int n)
        {
            vint32m2_t ix = __riscv_vmin(__riscv_vmax(__riscv_vfcvt_rtz_x(x, vl), 0, vl), n - 1, vl);
            x = __riscv_vfsub(x, __riscv_vfcvt_f(ix, vl), vl);
            ix = __riscv_vmul(ix, 4 * sizeof(float), vl);

            vfloat32m2x4_t val;
            val = __riscv_vloxseg4ei32_v_f32m2x4(tab, __riscv_vreinterpret_v_i32m2_u32m2(ix), vl);
            return __riscv_vfmadd(__riscv_vfmadd(__riscv_vfmadd(__riscv_vget_v_f32m2x4_f32m2(val, 3), x, __riscv_vget_v_f32m2x4_f32m2(val, 2), vl), x, __riscv_vget_v_f32m2x4_f32m2(val, 1), vl), x, __riscv_vget_v_f32m2x4_f32m2(val, 0), vl);
        }
    };
} // cv::cv_hal_rvv::LabTable

namespace LabtoBGR {
#undef cv_hal_cvtLabtoBGR
#define cv_hal_cvtLabtoBGR cv::cv_hal_rvv::LabtoBGR::cvtLabtoBGR

template<typename T>
static inline int cvtLabtoBGR(int start, int end, const T * src, size_t src_step, T * dst, size_t dst_step, int width, int dcn, bool swapBlue, bool isLab, bool srgb);

// the algorithm is copied from imgproc/src/color_lab.cpp,
// in the functor struct Lab2RGBfloat, Lab2RGBinteger, Luv2RGBfloat and Luv2RGBinteger
template<>
inline int cvtLabtoBGR<uchar>(int start, int end, const uchar * src, size_t src_step, uchar * dst, size_t dst_step, int width, int dcn, bool swapBlue, bool isLab, bool srgb)
{
    static const int XYZ2BGR[] =
    {
        (int)roundeven((1 << 12) *  0.055648f * 0.950456f), (int)roundeven((1 << 12) * -0.204043f), (int)roundeven((1 << 12) *  1.057311f * 1.088754f),
        (int)roundeven((1 << 12) * -0.969256f * 0.950456f), (int)roundeven((1 << 12) *  1.875991f), (int)roundeven((1 << 12) *  0.041556f * 1.088754f),
        (int)roundeven((1 << 12) *  3.240479f * 0.950456f), (int)roundeven((1 << 12) * -1.53715f ), (int)roundeven((1 << 12) * -0.498535f * 1.088754f)
    };
    static const int XYZ2BGR_D65[] =
    {
        (int)roundeven((1 << 12) *  0.055648f), (int)roundeven((1 << 12) * -0.204043f), (int)roundeven((1 << 12) *  1.057311f),
        (int)roundeven((1 << 12) * -0.969256f), (int)roundeven((1 << 12) *  1.875991f), (int)roundeven((1 << 12) *  0.041556f),
        (int)roundeven((1 << 12) *  3.240479f), (int)roundeven((1 << 12) * -1.53715f ), (int)roundeven((1 << 12) * -0.498535f)
    };

    const int* XYZtab = isLab ? XYZ2BGR : XYZ2BGR_D65;
    auto alpha = __riscv_vmv_v_x_u8m1(std::numeric_limits<uchar>::max(), __riscv_vsetvlmax_e8m1());
    for (int i = start; i < end; i++)
    {
        int vl;
        for (int j = 0; j < width; j += vl)
        {
            vl = __riscv_vsetvl_e8m1(width - j);
            auto vec_src = __riscv_vlseg3e8_v_u8m1x3(src + i * src_step + j * 3, vl);
            auto l = __riscv_vzext_vf4(__riscv_vget_v_u8m1x3_u8m1(vec_src, 0), vl);
            auto a = __riscv_vzext_vf4(__riscv_vget_v_u8m1x3_u8m1(vec_src, 1), vl);
            auto b = __riscv_vzext_vf4(__riscv_vget_v_u8m1x3_u8m1(vec_src, 2), vl);

            auto vec_yf = __riscv_vloxseg2ei32_v_i16m2x2(LabTable::Tab::instance().LabToYF_b, __riscv_vmul(l, 2 * sizeof(ushort), vl), vl);
            auto y = __riscv_vsext_vf2(__riscv_vget_v_i16m2x2_i16m2(vec_yf, 0), vl);

            vint32m4_t x, z;
            if (isLab)
            {
                auto ify = __riscv_vsext_vf2(__riscv_vget_v_i16m2x2_i16m2(vec_yf, 1), vl);
                auto adiv = __riscv_vsub(__riscv_vsra(__riscv_vmadd(__riscv_vreinterpret_v_u32m4_i32m4(a), 5*53687, __riscv_vmv_v_x_i32m4(1 << 7, vl), vl), 13, vl), 128*LabTable::Tab::LAB_BASE/500  , vl);
                auto bdiv = __riscv_vsub(__riscv_vsra(__riscv_vmadd(__riscv_vreinterpret_v_u32m4_i32m4(b),   41943, __riscv_vmv_v_x_i32m4(1 << 4, vl), vl),  9, vl), 128*LabTable::Tab::LAB_BASE/200-1, vl); // not +1 here

                auto fx = __riscv_vadd(ify, adiv, vl);
                auto fz = __riscv_vsub(ify, bdiv, vl);
                x = __riscv_vsub(__riscv_vdiv(__riscv_vmul(fx, 108, vl), 841, vl), LabTable::Tab::LAB_BASE*16/116*108/841, vl);
                z = __riscv_vsub(__riscv_vdiv(__riscv_vmul(fz, 108, vl), 841, vl), LabTable::Tab::LAB_BASE*16/116*108/841, vl);
                x = __riscv_vmerge(__riscv_vsra(__riscv_vmul(__riscv_vsra(__riscv_vmul(fx, fx, vl), 14, vl), fx, vl), 14, vl), x, __riscv_vmsle(fx, 3390, vl), vl);
                z = __riscv_vmerge(__riscv_vsra(__riscv_vmul(__riscv_vsra(__riscv_vmul(fz, fz, vl), 14, vl), fz, vl), 14, vl), z, __riscv_vmsle(fz, 3390, vl), vl);
            }
            else
            {
                auto up = __riscv_vloxei32_v_i32m4(LabTable::Tab::instance().LuToUp_b, __riscv_vmul(__riscv_vmadd(l, 256, a, vl), sizeof(int), vl), vl);
                auto vp = __riscv_vloxei32_v_i32m4(LabTable::Tab::instance().LvToVp_b, __riscv_vmul(__riscv_vmadd(l, 256, b, vl), sizeof(int), vl), vl);

                auto xv = __riscv_vwmul(up, vp, vl);
                x = __riscv_vncvt_x(__riscv_vsra(__riscv_vmul(__riscv_vsext_vf2(y, vl), __riscv_vsra(xv, 14, vl), vl), 14, vl), vl);

                auto vpl = __riscv_vmul(__riscv_vwmulsu(vp, l, vl), 15600*(LabTable::Tab::LAB_BASE/1024), vl);
                auto zp = __riscv_vsra(__riscv_vnmsub(xv, 255 / 3, vpl, vl), 14, vl);
                auto zq = __riscv_vsub(zp, 5 * 255 * LabTable::Tab::LAB_BASE, vl);
                auto zm = __riscv_vncvt_x(__riscv_vsra(__riscv_vmul(__riscv_vsext_vf2(y, vl), zq, vl), 14, vl), vl);
                z = __riscv_vadd(__riscv_vsra(zm, 8, vl), __riscv_vsra(zm, 16, vl), vl);

                x = __riscv_vmin(__riscv_vmax(x, 0, vl), 2 * LabTable::Tab::LAB_BASE, vl);
                z = __riscv_vmin(__riscv_vmax(z, 0, vl), 2 * LabTable::Tab::LAB_BASE, vl);
            }

            auto bo = __riscv_vssra(__riscv_vmadd(x, XYZtab[0], __riscv_vmadd(y, XYZtab[1], __riscv_vmul(z, XYZtab[2], vl), vl), vl), 14, __RISCV_VXRM_RNU, vl);
            auto go = __riscv_vssra(__riscv_vmadd(x, XYZtab[3], __riscv_vmadd(y, XYZtab[4], __riscv_vmul(z, XYZtab[5], vl), vl), vl), 14, __RISCV_VXRM_RNU, vl);
            auto ro = __riscv_vssra(__riscv_vmadd(x, XYZtab[6], __riscv_vmadd(y, XYZtab[7], __riscv_vmul(z, XYZtab[8], vl), vl), vl), 14, __RISCV_VXRM_RNU, vl);
            bo = __riscv_vmin(__riscv_vmax(bo, 0, vl), LabTable::Tab::INV_GAMMA_TAB_SIZE - 1, vl);
            go = __riscv_vmin(__riscv_vmax(go, 0, vl), LabTable::Tab::INV_GAMMA_TAB_SIZE - 1, vl);
            ro = __riscv_vmin(__riscv_vmax(ro, 0, vl), LabTable::Tab::INV_GAMMA_TAB_SIZE - 1, vl);
            vuint16m2_t bb, gg, rr;
            if (srgb)
            {
                bb = __riscv_vloxei16_v_u16m2(LabTable::Tab::instance().sRGBInvGammaTab_b, __riscv_vmul(__riscv_vncvt_x(__riscv_vreinterpret_v_i32m4_u32m4(bo), vl), sizeof(ushort), vl), vl);
                gg = __riscv_vloxei16_v_u16m2(LabTable::Tab::instance().sRGBInvGammaTab_b, __riscv_vmul(__riscv_vncvt_x(__riscv_vreinterpret_v_i32m4_u32m4(go), vl), sizeof(ushort), vl), vl);
                rr = __riscv_vloxei16_v_u16m2(LabTable::Tab::instance().sRGBInvGammaTab_b, __riscv_vmul(__riscv_vncvt_x(__riscv_vreinterpret_v_i32m4_u32m4(ro), vl), sizeof(ushort), vl), vl);
            }
            else
            {
                bb = __riscv_vnclipu(__riscv_vreinterpret_v_i32m4_u32m4(__riscv_vsub(__riscv_vsll(bo, 8, vl), bo, vl)), 12, __RISCV_VXRM_RDN, vl);
                gg = __riscv_vnclipu(__riscv_vreinterpret_v_i32m4_u32m4(__riscv_vsub(__riscv_vsll(go, 8, vl), go, vl)), 12, __RISCV_VXRM_RDN, vl);
                rr = __riscv_vnclipu(__riscv_vreinterpret_v_i32m4_u32m4(__riscv_vsub(__riscv_vsll(ro, 8, vl), ro, vl)), 12, __RISCV_VXRM_RDN, vl);
            }

            if (swapBlue)
            {
                auto t = bb;
                bb = rr, rr = t;
            }
            if (dcn == 3)
            {
                vuint8m1x3_t vec_dst{};
                vec_dst = __riscv_vset_v_u8m1_u8m1x3(vec_dst, 0, __riscv_vnclipu(bb, 0, __RISCV_VXRM_RNU, vl));
                vec_dst = __riscv_vset_v_u8m1_u8m1x3(vec_dst, 1, __riscv_vnclipu(gg, 0, __RISCV_VXRM_RNU, vl));
                vec_dst = __riscv_vset_v_u8m1_u8m1x3(vec_dst, 2, __riscv_vnclipu(rr, 0, __RISCV_VXRM_RNU, vl));
                __riscv_vsseg3e8(dst + i * dst_step + j * 3, vec_dst, vl);
            }
            else
            {
                vuint8m1x4_t vec_dst{};
                vec_dst = __riscv_vset_v_u8m1_u8m1x4(vec_dst, 0, __riscv_vnclipu(bb, 0, __RISCV_VXRM_RNU, vl));
                vec_dst = __riscv_vset_v_u8m1_u8m1x4(vec_dst, 1, __riscv_vnclipu(gg, 0, __RISCV_VXRM_RNU, vl));
                vec_dst = __riscv_vset_v_u8m1_u8m1x4(vec_dst, 2, __riscv_vnclipu(rr, 0, __RISCV_VXRM_RNU, vl));
                vec_dst = __riscv_vset_v_u8m1_u8m1x4(vec_dst, 3, alpha);
                __riscv_vsseg4e8(dst + i * dst_step + j * 4, vec_dst, vl);
            }
        }
    }

    return CV_HAL_ERROR_OK;
}

template<>
inline int cvtLabtoBGR<float>(int start, int end, const float * src, size_t src_step, float * dst, size_t dst_step, int width, int dcn, bool swapBlue, bool isLab, bool srgb)
{
    static constexpr float XYZ2BGR[] =
    {
         0.055648f * 0.950456f, -0.204043f,  1.057311f * 1.088754f,
        -0.969256f * 0.950456f,  1.875991f,  0.041556f * 1.088754f,
         3.240479f * 0.950456f, -1.53715f , -0.498535f * 1.088754f
    };
    static constexpr float XYZ2BGR_D65[] =
    {
         0.055648f, -0.204043f,  1.057311f,
        -0.969256f,  1.875991f,  0.041556f,
         3.240479f, -1.53715f , -0.498535f
    };

    src_step /= sizeof(float);
    dst_step /= sizeof(float);

    const float* XYZtab = isLab ? XYZ2BGR : XYZ2BGR_D65;
    auto alpha = __riscv_vfmv_v_f_f32m2(1.0f, __riscv_vsetvlmax_e32m2());
    for (int i = start; i < end; i++)
    {
        int vl;
        for (int j = 0; j < width; j += vl)
        {
            vl = __riscv_vsetvl_e32m2(width - j);
            auto vec_src = __riscv_vlseg3e32_v_f32m2x3(src + i * src_step + j * 3, vl);
            auto l = __riscv_vget_v_f32m2x3_f32m2(vec_src, 0), a = __riscv_vget_v_f32m2x3_f32m2(vec_src, 1), b = __riscv_vget_v_f32m2x3_f32m2(vec_src, 2);

            auto y = __riscv_vfmul(l, 1.0f / 903.3f, vl);
            auto fy = __riscv_vfmul(__riscv_vfadd(l, 16.0f, vl), 1.0f / 116.0f, vl);

            vfloat32m2_t x, z;
            if (isLab)
            {
                fy = __riscv_vmerge(fy, __riscv_vfmadd(y, 7.787f, __riscv_vfmv_v_f_f32m2(16.0f / 116.0f, vl), vl), __riscv_vmfle(l, 8.0f, vl), vl);
                y = __riscv_vmerge(y, __riscv_vfmul(__riscv_vfmul(fy, fy, vl), fy, vl), __riscv_vmfgt(l, 8.0f, vl), vl);

                x = __riscv_vfmadd(a, 1.0f / 500.0f, fy, vl);
                z = __riscv_vfmadd(b, -1.0f / 200.0f, fy, vl);
                x = __riscv_vmerge(__riscv_vfmul(__riscv_vfmul(x, x, vl), x, vl), __riscv_vfmul(__riscv_vfsub(x, 16.0f / 116.0f, vl), 1.0f / 7.787f, vl), __riscv_vmfle(x, 6.0f / 29.0f, vl), vl);
                z = __riscv_vmerge(__riscv_vfmul(__riscv_vfmul(z, z, vl), z, vl), __riscv_vfmul(__riscv_vfsub(z, 16.0f / 116.0f, vl), 1.0f / 7.787f, vl), __riscv_vmfle(z, 6.0f / 29.0f, vl), vl);
            }
            else
            {
                y = __riscv_vmerge(y, __riscv_vfmul(__riscv_vfmul(fy, fy, vl), fy, vl), __riscv_vmfgt(l, 8.0f, vl), vl);
                auto up = __riscv_vfmul (__riscv_vfmadd(l, 2.5719122887f, a, vl),  3.0f, vl);
                auto vp = __riscv_vfrdiv(__riscv_vfmadd(l, 6.0884485245f, b, vl), 0.25f, vl);
                vp = __riscv_vfmin(__riscv_vfmax(vp, -0.25f, vl), 0.25f, vl);
                x = __riscv_vfmul(__riscv_vfmul(__riscv_vfmul(up, vp, vl), 3.0f, vl), y, vl);
                z = __riscv_vfmul(__riscv_vfmsub(__riscv_vfmsub(l, 156.0f, up, vl), vp, __riscv_vfmv_v_f_f32m2(5.0f, vl), vl), y, vl);
            }

            auto bo = __riscv_vfmadd(x, XYZtab[0], __riscv_vfmadd(y, XYZtab[1], __riscv_vfmul(z, XYZtab[2], vl), vl), vl);
            auto go = __riscv_vfmadd(x, XYZtab[3], __riscv_vfmadd(y, XYZtab[4], __riscv_vfmul(z, XYZtab[5], vl), vl), vl);
            auto ro = __riscv_vfmadd(x, XYZtab[6], __riscv_vfmadd(y, XYZtab[7], __riscv_vfmul(z, XYZtab[8], vl), vl), vl);
            bo = __riscv_vfmin(__riscv_vfmax(bo, 0.0f, vl), 1.0f, vl);
            go = __riscv_vfmin(__riscv_vfmax(go, 0.0f, vl), 1.0f, vl);
            ro = __riscv_vfmin(__riscv_vfmax(ro, 0.0f, vl), 1.0f, vl);
            if (srgb)
            {
                bo = LabTable::Tab::splineInterpolate(vl, __riscv_vfmul(bo, LabTable::Tab::GAMMA_TAB_SIZE, vl), LabTable::Tab::instance().sRGBInvGammaTab, LabTable::Tab::GAMMA_TAB_SIZE);
                go = LabTable::Tab::splineInterpolate(vl, __riscv_vfmul(go, LabTable::Tab::GAMMA_TAB_SIZE, vl), LabTable::Tab::instance().sRGBInvGammaTab, LabTable::Tab::GAMMA_TAB_SIZE);
                ro = LabTable::Tab::splineInterpolate(vl, __riscv_vfmul(ro, LabTable::Tab::GAMMA_TAB_SIZE, vl), LabTable::Tab::instance().sRGBInvGammaTab, LabTable::Tab::GAMMA_TAB_SIZE);
            }

            if (swapBlue)
            {
                auto t = bo;
                bo = ro, ro = t;
            }
            if (dcn == 3)
            {
                vfloat32m2x3_t vec_dst{};
                vec_dst = __riscv_vset_v_f32m2_f32m2x3(vec_dst, 0, bo);
                vec_dst = __riscv_vset_v_f32m2_f32m2x3(vec_dst, 1, go);
                vec_dst = __riscv_vset_v_f32m2_f32m2x3(vec_dst, 2, ro);
                __riscv_vsseg3e32(dst + i * dst_step + j * 3, vec_dst, vl);
            }
            else
            {
                vfloat32m2x4_t vec_dst{};
                vec_dst = __riscv_vset_v_f32m2_f32m2x4(vec_dst, 0, bo);
                vec_dst = __riscv_vset_v_f32m2_f32m2x4(vec_dst, 1, go);
                vec_dst = __riscv_vset_v_f32m2_f32m2x4(vec_dst, 2, ro);
                vec_dst = __riscv_vset_v_f32m2_f32m2x4(vec_dst, 3, alpha);
                __riscv_vsseg4e32(dst + i * dst_step + j * 4, vec_dst, vl);
            }
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int cvtLabtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int depth, int dcn, bool swapBlue, bool isLab, bool srgb)
{
    if (dcn != 3 && dcn != 4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    switch (depth)
    {
    case CV_8U:
        return color::invoke(width, height, {cvtLabtoBGR<uchar>}, reinterpret_cast<const uchar*>(src_data), src_step, reinterpret_cast<uchar*>(dst_data), dst_step, width, dcn, swapBlue, isLab, srgb);
    case CV_32F:
        return color::invoke(width, height, {cvtLabtoBGR<float>}, reinterpret_cast<const float*>(src_data), src_step, reinterpret_cast<float*>(dst_data), dst_step, width, dcn, swapBlue, isLab, srgb);
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}
} // cv::cv_hal_rvv::LabtoBGR

namespace BGRtoLab {
#undef cv_hal_cvtBGRtoLab
#define cv_hal_cvtBGRtoLab cv::cv_hal_rvv::BGRtoLab::cvtBGRtoLab

struct rvv_base
{
    using T = vuint16m2_t;
    using S = vint16m2_t;
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e8m1(a); }
    static inline void vlseg(const uchar* a, int b, T& c, T& d, T& e, size_t f)
    {
        if (b == 3)
        {
            auto x = __riscv_vlseg3e8_v_u8m1x3(a, f);
            c = __riscv_vzext_vf2(__riscv_vget_v_u8m1x3_u8m1(x, 0), f);
            d = __riscv_vzext_vf2(__riscv_vget_v_u8m1x3_u8m1(x, 1), f);
            e = __riscv_vzext_vf2(__riscv_vget_v_u8m1x3_u8m1(x, 2), f);
        }
        else
        {
            auto x = __riscv_vlseg4e8_v_u8m1x4(a, f);
            c = __riscv_vzext_vf2(__riscv_vget_v_u8m1x4_u8m1(x, 0), f);
            d = __riscv_vzext_vf2(__riscv_vget_v_u8m1x4_u8m1(x, 1), f);
            e = __riscv_vzext_vf2(__riscv_vget_v_u8m1x4_u8m1(x, 2), f);
        }
    }
    static inline void vsseg(uchar* a, S b, S c, S d, size_t e)
    {
        vuint8m1x3_t x{};
        x = __riscv_vset_v_u8m1_u8m1x3(x, 0, __riscv_vnclipu(__riscv_vreinterpret_v_i16m2_u16m2(__riscv_vmax(b, 0, e)), 0, __RISCV_VXRM_RNU, e));
        x = __riscv_vset_v_u8m1_u8m1x3(x, 1, __riscv_vnclipu(__riscv_vreinterpret_v_i16m2_u16m2(__riscv_vmax(c, 0, e)), 0, __RISCV_VXRM_RNU, e));
        x = __riscv_vset_v_u8m1_u8m1x3(x, 2, __riscv_vnclipu(__riscv_vreinterpret_v_i16m2_u16m2(__riscv_vmax(d, 0, e)), 0, __RISCV_VXRM_RNU, e));
        __riscv_vsseg3e8(a, x, e);
    }
};

template<bool isLab, bool srgb> struct rvv;
template<bool srgb> struct rvv<true, srgb> : rvv_base
{
    static inline void process(T b, T g, T r, S& lo, S& ao, S& bo, int vl)
    {
        static const ushort BGR2XYZ[] =
        {
            (ushort)roundeven((1 << 12) * 0.180423f / 0.950456f), (ushort)roundeven((1 << 12) * 0.357580f / 0.950456f), (ushort)roundeven((1 << 12) * 0.412453f / 0.950456f),
            (ushort)roundeven((1 << 12) * 0.072169f            ), (ushort)roundeven((1 << 12) * 0.715160f            ), (ushort)roundeven((1 << 12) * 0.212671f            ),
            (ushort)roundeven((1 << 12) * 0.950227f / 1.088754f), (ushort)roundeven((1 << 12) * 0.119193f / 1.088754f), (ushort)roundeven((1 << 12) * 0.019334f / 1.088754f)
        };

        vuint16m2_t bb, gg, rr;
        if (srgb)
        {
            bb = __riscv_vloxei16_v_u16m2(LabTable::Tab::instance().sRGBGammaTab_b, __riscv_vmul(b, sizeof(ushort), vl), vl);
            gg = __riscv_vloxei16_v_u16m2(LabTable::Tab::instance().sRGBGammaTab_b, __riscv_vmul(g, sizeof(ushort), vl), vl);
            rr = __riscv_vloxei16_v_u16m2(LabTable::Tab::instance().sRGBGammaTab_b, __riscv_vmul(r, sizeof(ushort), vl), vl);
        }
        else
        {
            bb = __riscv_vsll(b, 3, vl);
            gg = __riscv_vsll(g, 3, vl);
            rr = __riscv_vsll(r, 3, vl);
        }

        auto x = __riscv_vnclipu(__riscv_vwmaccu(__riscv_vwmaccu(__riscv_vwmulu(bb, BGR2XYZ[0], vl), BGR2XYZ[1], gg, vl), BGR2XYZ[2], rr, vl), 12, __RISCV_VXRM_RNU, vl);
        auto y = __riscv_vnclipu(__riscv_vwmaccu(__riscv_vwmaccu(__riscv_vwmulu(bb, BGR2XYZ[3], vl), BGR2XYZ[4], gg, vl), BGR2XYZ[5], rr, vl), 12, __RISCV_VXRM_RNU, vl);
        auto z = __riscv_vnclipu(__riscv_vwmaccu(__riscv_vwmaccu(__riscv_vwmulu(bb, BGR2XYZ[6], vl), BGR2XYZ[7], gg, vl), BGR2XYZ[8], rr, vl), 12, __RISCV_VXRM_RNU, vl);
        x = __riscv_vloxei16_v_u16m2(LabTable::Tab::instance().LabCbrtTab_b, __riscv_vmul(x, sizeof(ushort), vl), vl);
        y = __riscv_vloxei16_v_u16m2(LabTable::Tab::instance().LabCbrtTab_b, __riscv_vmul(y, sizeof(ushort), vl), vl);
        z = __riscv_vloxei16_v_u16m2(LabTable::Tab::instance().LabCbrtTab_b, __riscv_vmul(z, sizeof(ushort), vl), vl);

        lo = __riscv_vnclip(__riscv_vwmaccsu(__riscv_vmv_v_x_i32m4(-1336934, vl), 296, y, vl), 15, __RISCV_VXRM_RNU, vl);
        ao = __riscv_vnclip(__riscv_vwmacc(__riscv_vmv_v_x_i32m4(128 << 15, vl), 500, __riscv_vsub(__riscv_vreinterpret_v_u16m2_i16m2(x), __riscv_vreinterpret_v_u16m2_i16m2(y), vl), vl), 15, __RISCV_VXRM_RNU, vl);
        bo = __riscv_vnclip(__riscv_vwmacc(__riscv_vmv_v_x_i32m4(128 << 15, vl), 200, __riscv_vsub(__riscv_vreinterpret_v_u16m2_i16m2(y), __riscv_vreinterpret_v_u16m2_i16m2(z), vl), vl), 15, __RISCV_VXRM_RNU, vl);
    }
};
template<> struct rvv<false, true> : rvv_base
{
    static inline void process(T b, T g, T r, S& lo, S& ao, S& bo, int vl)
    {
        auto x = __riscv_vand(__riscv_vsll(b, 1, vl), 15, vl), y = __riscv_vand(__riscv_vsll(g, 1, vl), 15, vl), z = __riscv_vand(__riscv_vsll(r, 1, vl), 15, vl);
        auto base = __riscv_vmul(__riscv_vwmaccu(__riscv_vwmaccu(__riscv_vwmulu(x, 8, vl), 8*LabTable::Tab::TRILINEAR_BASE, y, vl), 8*LabTable::Tab::TRILINEAR_BASE*LabTable::Tab::TRILINEAR_BASE, z, vl), sizeof(short), vl);
        auto tab = __riscv_vloxseg4ei32_v_i16m2x4(LabTable::Tab::instance().trilinearLUT, base, vl);
        auto w0 = __riscv_vget_v_i16m2x4_i16m2(tab, 0);
        auto w1 = __riscv_vget_v_i16m2x4_i16m2(tab, 1);
        auto w2 = __riscv_vget_v_i16m2x4_i16m2(tab, 2);
        auto w3 = __riscv_vget_v_i16m2x4_i16m2(tab, 3);
        tab = __riscv_vloxseg4ei32_v_i16m2x4(LabTable::Tab::instance().trilinearLUT, __riscv_vadd(base, 4 * sizeof(short), vl), vl);
        auto w4 = __riscv_vget_v_i16m2x4_i16m2(tab, 0);
        auto w5 = __riscv_vget_v_i16m2x4_i16m2(tab, 1);
        auto w6 = __riscv_vget_v_i16m2x4_i16m2(tab, 2);
        auto w7 = __riscv_vget_v_i16m2x4_i16m2(tab, 3);

        auto tx = __riscv_vsrl(b, 3, vl), ty = __riscv_vsrl(g, 3, vl), tz = __riscv_vsrl(r, 3, vl);
        base = __riscv_vmul(__riscv_vwmaccu(__riscv_vwmaccu(__riscv_vwmulu(tx, 3*8, vl), 3*8*LabTable::Tab::LAB_LUT_DIM, ty, vl), 3*8*LabTable::Tab::LAB_LUT_DIM*LabTable::Tab::LAB_LUT_DIM, tz, vl), sizeof(short), vl);
        auto interpolate = [&](vuint32m4_t p) {
            tab = __riscv_vloxseg4ei32_v_i16m2x4(LabTable::Tab::instance().RGB2LuvLUT, p, vl);
            auto a0 = __riscv_vget_v_i16m2x4_i16m2(tab, 0);
            auto a1 = __riscv_vget_v_i16m2x4_i16m2(tab, 1);
            auto a2 = __riscv_vget_v_i16m2x4_i16m2(tab, 2);
            auto a3 = __riscv_vget_v_i16m2x4_i16m2(tab, 3);
            tab = __riscv_vloxseg4ei32_v_i16m2x4(LabTable::Tab::instance().RGB2LuvLUT, __riscv_vadd(p, 4 * sizeof(short), vl), vl);
            auto a4 = __riscv_vget_v_i16m2x4_i16m2(tab, 0);
            auto a5 = __riscv_vget_v_i16m2x4_i16m2(tab, 1);
            auto a6 = __riscv_vget_v_i16m2x4_i16m2(tab, 2);
            auto a7 = __riscv_vget_v_i16m2x4_i16m2(tab, 3);
            return __riscv_vwmacc(__riscv_vwmacc(__riscv_vwmacc(__riscv_vwmacc(__riscv_vwmacc(__riscv_vwmacc(__riscv_vwmacc(__riscv_vwmul(a0, w0, vl), a1, w1, vl), a2, w2, vl), a3, w3, vl), a4, w4, vl), a5, w5, vl), a6, w6, vl), a7, w7, vl);
        };

        lo = __riscv_vnclip(__riscv_vssra(interpolate(base), 12, __RISCV_VXRM_RNU, vl), 6, __RISCV_VXRM_RDN, vl);
        ao = __riscv_vnclip(__riscv_vssra(interpolate(__riscv_vadd(base, 8 * sizeof(short), vl)), 12, __RISCV_VXRM_RNU, vl), 6, __RISCV_VXRM_RDN, vl);
        bo = __riscv_vnclip(__riscv_vssra(interpolate(__riscv_vadd(base, 16 * sizeof(short), vl)), 12, __RISCV_VXRM_RNU, vl), 6, __RISCV_VXRM_RDN, vl);
    }
};
template<> struct rvv<false, false>
{
    using T = vint32m2_t;
    using S = vfloat32m2_t;
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e8mf2(a); }
    static inline void vlseg(const uchar* a, int b, T& c, T& d, T& e, size_t f)
    {
        if (b == 3)
        {
            auto x = __riscv_vlseg3e8_v_u8mf2x3(a, f);
            c = __riscv_vreinterpret_v_u32m2_i32m2(__riscv_vzext_vf4(__riscv_vget_v_u8mf2x3_u8mf2(x, 0), f));
            d = __riscv_vreinterpret_v_u32m2_i32m2(__riscv_vzext_vf4(__riscv_vget_v_u8mf2x3_u8mf2(x, 1), f));
            e = __riscv_vreinterpret_v_u32m2_i32m2(__riscv_vzext_vf4(__riscv_vget_v_u8mf2x3_u8mf2(x, 2), f));
        }
        else
        {
            auto x = __riscv_vlseg4e8_v_u8mf2x4(a, f);
            c = __riscv_vreinterpret_v_u32m2_i32m2(__riscv_vzext_vf4(__riscv_vget_v_u8mf2x4_u8mf2(x, 0), f));
            d = __riscv_vreinterpret_v_u32m2_i32m2(__riscv_vzext_vf4(__riscv_vget_v_u8mf2x4_u8mf2(x, 1), f));
            e = __riscv_vreinterpret_v_u32m2_i32m2(__riscv_vzext_vf4(__riscv_vget_v_u8mf2x4_u8mf2(x, 2), f));
        }
    }
    static inline void vsseg(uchar* a, S b, S c, S d, size_t e)
    {
        vfloat32m2x3_t x{};
        x = __riscv_vset_v_f32m2_f32m2x3(x, 0, b);
        x = __riscv_vset_v_f32m2_f32m2x3(x, 1, c);
        x = __riscv_vset_v_f32m2_f32m2x3(x, 2, d);
        __riscv_vsseg3e32(reinterpret_cast<float*>(a), x, e);
    }
    static inline void process(T b, T g, T r, S& lo, S& ao, S& bo, int vl)
    {
        lo = __riscv_vfmul(__riscv_vfcvt_f(b, vl), 1.0f / 255.0f, vl);
        ao = __riscv_vfmul(__riscv_vfcvt_f(g, vl), 1.0f / 255.0f, vl);
        bo = __riscv_vfmul(__riscv_vfcvt_f(r, vl), 1.0f / 255.0f, vl);
    }
};

static inline int cvtBGRtoLab_f(int, int, const float *, size_t, float *, size_t, int, int, bool, bool, bool);

// the algorithm is copied from imgproc/src/color_lab.cpp,
// in the functor struct RGB2Lab_f, RGB2Lab_b, RGB2Luv_b, RGB2Luvfloat and RGB2Luvinterpolate
template<bool isLab, bool srgb>
static inline int cvtBGRtoLab_u(int start, int end, const uchar * src, size_t src_step, uchar * dst, size_t dst_step, int width, int scn, bool swapBlue)
{
    std::vector<float> buf(width * 3);
    for (int i = start; i < end; i++)
    {
        int vl;
        for (int j = 0; j < width; j += vl)
        {
            vl = rvv<isLab, srgb>::vsetvl(width - j);
            typename rvv<isLab, srgb>::T b, g, r;
            rvv<isLab, srgb>::vlseg(src + i * src_step + j * scn, scn, b, g, r, vl);
            if (swapBlue)
            {
                auto t = b;
                b = r, r = t;
            }

            typename rvv<isLab, srgb>::S lo, ao, bo;
            rvv<isLab, srgb>::process(b, g, r, lo, ao, bo, vl);
            rvv<isLab, srgb>::vsseg(isLab || srgb ? dst + i * dst_step + j * 3 : (uchar*)(buf.data() + j * 3), lo, ao, bo, vl);
        }

        if (!isLab && !srgb)
        {
            cvtBGRtoLab_f(0, 1, buf.data(), sizeof(float), buf.data(), sizeof(float), width, 3, false, false, false);
            for (int j = 0; j < width; j += vl)
            {
                vl = __riscv_vsetvl_e32m2(width - j);
                auto vec_src = __riscv_vlseg3e32_v_f32m2x3(buf.data() + j * 3, vl);
                auto l = __riscv_vget_v_f32m2x3_f32m2(vec_src, 0), a = __riscv_vget_v_f32m2x3_f32m2(vec_src, 1), b = __riscv_vget_v_f32m2x3_f32m2(vec_src, 2);

                auto lo = __riscv_vfmul(l, 2.55f, vl);
                auto ao = __riscv_vfmadd(a, 255.0f/354.0f, __riscv_vfmv_v_f_f32m2(134.0f*255.0f/354.0f, vl), vl);
                auto bo = __riscv_vfmadd(b, 255.0f/262.0f, __riscv_vfmv_v_f_f32m2(140.0f*255.0f/262.0f, vl), vl);

                vuint8mf2x3_t vec_dst{};
                vec_dst = __riscv_vset_v_u8mf2_u8mf2x3(vec_dst, 0, __riscv_vnclipu(__riscv_vfncvt_xu(lo, vl), 0, __RISCV_VXRM_RNU, vl));
                vec_dst = __riscv_vset_v_u8mf2_u8mf2x3(vec_dst, 1, __riscv_vnclipu(__riscv_vfncvt_xu(ao, vl), 0, __RISCV_VXRM_RNU, vl));
                vec_dst = __riscv_vset_v_u8mf2_u8mf2x3(vec_dst, 2, __riscv_vnclipu(__riscv_vfncvt_xu(bo, vl), 0, __RISCV_VXRM_RNU, vl));
                __riscv_vsseg3e8(dst + i * dst_step + j * 3, vec_dst, vl);
            }
        }
    }

    return CV_HAL_ERROR_OK;
}

static inline int cvtBGRtoLab_f(int start, int end, const float * src, size_t src_step, float * dst, size_t dst_step, int width, int scn, bool swapBlue, bool isLab, bool srgb)
{
    static constexpr float BGR2XYZ[] =
    {
        0.180423f / 0.950456f, 0.357580f / 0.950456f, 0.412453f / 0.950456f,
        0.072169f            , 0.715160f            , 0.212671f            ,
        0.950227f / 1.088754f, 0.119193f / 1.088754f, 0.019334f / 1.088754f
    };
    static constexpr float BGR2XYZ_D65[] =
    {
        0.180423f, 0.357580f, 0.412453f,
        0.072169f, 0.715160f, 0.212671f,
        0.950227f, 0.119193f, 0.019334f
    };

    src_step /= sizeof(float);
    dst_step /= sizeof(float);

    const float* BGRtab = isLab ? BGR2XYZ : BGR2XYZ_D65;
    for (int i = start; i < end; i++)
    {
        int vl;
        for (int j = 0; j < width; j += vl)
        {
            vl = __riscv_vsetvl_e32m2(width - j);
            vfloat32m2_t b, g, r;
            if (scn == 3)
            {
                auto vec_src = __riscv_vlseg3e32_v_f32m2x3(src + i * src_step + j * 3, vl);
                b = __riscv_vget_v_f32m2x3_f32m2(vec_src, 0);
                g = __riscv_vget_v_f32m2x3_f32m2(vec_src, 1);
                r = __riscv_vget_v_f32m2x3_f32m2(vec_src, 2);
            }
            else
            {
                auto vec_src = __riscv_vlseg4e32_v_f32m2x4(src + i * src_step + j * 4, vl);
                b = __riscv_vget_v_f32m2x4_f32m2(vec_src, 0);
                g = __riscv_vget_v_f32m2x4_f32m2(vec_src, 1);
                r = __riscv_vget_v_f32m2x4_f32m2(vec_src, 2);
            }
            if (swapBlue)
            {
                auto t = b;
                b = r, r = t;
            }
            b = __riscv_vfmin(__riscv_vfmax(b, 0.0f, vl), 1.0f, vl);
            g = __riscv_vfmin(__riscv_vfmax(g, 0.0f, vl), 1.0f, vl);
            r = __riscv_vfmin(__riscv_vfmax(r, 0.0f, vl), 1.0f, vl);

            vfloat32m2_t lo, ao, bo;
            if (isLab && srgb)
            {
                auto ib = __riscv_vfcvt_xu(__riscv_vfmul(b, LabTable::Tab::LAB_BASE, vl), vl);
                auto ig = __riscv_vfcvt_xu(__riscv_vfmul(g, LabTable::Tab::LAB_BASE, vl), vl);
                auto ir = __riscv_vfcvt_xu(__riscv_vfmul(r, LabTable::Tab::LAB_BASE, vl), vl);

                auto x = __riscv_vand(__riscv_vsrl(ib, 5, vl), 15, vl), y = __riscv_vand(__riscv_vsrl(ig, 5, vl), 15, vl), z = __riscv_vand(__riscv_vsrl(ir, 5, vl), 15, vl);
                auto base = __riscv_vmul(__riscv_vmacc(__riscv_vmacc(__riscv_vmul(x, 8, vl), 8*LabTable::Tab::TRILINEAR_BASE, y, vl), 8*LabTable::Tab::TRILINEAR_BASE*LabTable::Tab::TRILINEAR_BASE, z, vl), sizeof(short), vl);
                auto tab = __riscv_vloxseg8ei32_v_i16m1x8(LabTable::Tab::instance().trilinearLUT, base, vl);
                auto w0 = __riscv_vget_v_i16m1x8_i16m1(tab, 0);
                auto w1 = __riscv_vget_v_i16m1x8_i16m1(tab, 1);
                auto w2 = __riscv_vget_v_i16m1x8_i16m1(tab, 2);
                auto w3 = __riscv_vget_v_i16m1x8_i16m1(tab, 3);
                auto w4 = __riscv_vget_v_i16m1x8_i16m1(tab, 4);
                auto w5 = __riscv_vget_v_i16m1x8_i16m1(tab, 5);
                auto w6 = __riscv_vget_v_i16m1x8_i16m1(tab, 6);
                auto w7 = __riscv_vget_v_i16m1x8_i16m1(tab, 7);

                auto tx = __riscv_vsrl(ib, 9, vl), ty = __riscv_vsrl(ig, 9, vl), tz = __riscv_vsrl(ir, 9, vl);
                base = __riscv_vmul(__riscv_vmacc(__riscv_vmacc(__riscv_vmul(tx, 3*8, vl), 3*8*LabTable::Tab::LAB_LUT_DIM, ty, vl), 3*8*LabTable::Tab::LAB_LUT_DIM*LabTable::Tab::LAB_LUT_DIM, tz, vl), sizeof(short), vl);
                auto interpolate = [&](vuint32m2_t p) {
                    tab = __riscv_vloxseg8ei32_v_i16m1x8(LabTable::Tab::instance().RGB2LabLUT, p, vl);
                    auto a0 = __riscv_vget_v_i16m1x8_i16m1(tab, 0);
                    auto a1 = __riscv_vget_v_i16m1x8_i16m1(tab, 1);
                    auto a2 = __riscv_vget_v_i16m1x8_i16m1(tab, 2);
                    auto a3 = __riscv_vget_v_i16m1x8_i16m1(tab, 3);
                    auto a4 = __riscv_vget_v_i16m1x8_i16m1(tab, 4);
                    auto a5 = __riscv_vget_v_i16m1x8_i16m1(tab, 5);
                    auto a6 = __riscv_vget_v_i16m1x8_i16m1(tab, 6);
                    auto a7 = __riscv_vget_v_i16m1x8_i16m1(tab, 7);
                    return __riscv_vwmacc(__riscv_vwmacc(__riscv_vwmacc(__riscv_vwmacc(__riscv_vwmacc(__riscv_vwmacc(__riscv_vwmacc(__riscv_vwmul(a0, w0, vl), a1, w1, vl), a2, w2, vl), a3, w3, vl), a4, w4, vl), a5, w5, vl), a6, w6, vl), a7, w7, vl);
                };

                lo = __riscv_vfmul(__riscv_vfcvt_f(__riscv_vssra(interpolate(base), 12, __RISCV_VXRM_RNU, vl), vl), 100.0f / LabTable::Tab::LAB_BASE, vl);
                ao = __riscv_vfmadd(__riscv_vfcvt_f(__riscv_vssra(interpolate(__riscv_vadd(base, 8 * sizeof(short), vl)), 12, __RISCV_VXRM_RNU, vl), vl), 256.0f / LabTable::Tab::LAB_BASE, __riscv_vfmv_v_f_f32m2(-128.0f, vl), vl);
                bo = __riscv_vfmadd(__riscv_vfcvt_f(__riscv_vssra(interpolate(__riscv_vadd(base, 16 * sizeof(short), vl)), 12, __RISCV_VXRM_RNU, vl), vl), 256.0f / LabTable::Tab::LAB_BASE, __riscv_vfmv_v_f_f32m2(-128.0f, vl), vl);
            }
            else
            {
                if (srgb)
                {
                    b = LabTable::Tab::splineInterpolate(vl, __riscv_vfmul(b, LabTable::Tab::GAMMA_TAB_SIZE, vl), LabTable::Tab::instance().sRGBGammaTab, LabTable::Tab::GAMMA_TAB_SIZE);
                    g = LabTable::Tab::splineInterpolate(vl, __riscv_vfmul(g, LabTable::Tab::GAMMA_TAB_SIZE, vl), LabTable::Tab::instance().sRGBGammaTab, LabTable::Tab::GAMMA_TAB_SIZE);
                    r = LabTable::Tab::splineInterpolate(vl, __riscv_vfmul(r, LabTable::Tab::GAMMA_TAB_SIZE, vl), LabTable::Tab::instance().sRGBGammaTab, LabTable::Tab::GAMMA_TAB_SIZE);
                }

                auto x = __riscv_vfmadd(b, BGRtab[0], __riscv_vfmadd(g, BGRtab[1], __riscv_vfmul(r, BGRtab[2], vl), vl), vl);
                auto y = __riscv_vfmadd(b, BGRtab[3], __riscv_vfmadd(g, BGRtab[4], __riscv_vfmul(r, BGRtab[5], vl), vl), vl);
                auto z = __riscv_vfmadd(b, BGRtab[6], __riscv_vfmadd(g, BGRtab[7], __riscv_vfmul(r, BGRtab[8], vl), vl), vl);
                auto fy = LabTable::Tab::splineInterpolate(vl, __riscv_vfmul(y, LabTable::Tab::GAMMA_TAB_SIZE, vl), LabTable::Tab::instance().LabCbrtTab, LabTable::Tab::GAMMA_TAB_SIZE);

                lo = __riscv_vfmadd(fy, 116.0f, __riscv_vfmv_v_f_f32m2(-16.0f, vl), vl);
                if (isLab)
                {
                    x = LabTable::Tab::splineInterpolate(vl, __riscv_vfmul(x, LabTable::Tab::GAMMA_TAB_SIZE, vl), LabTable::Tab::instance().LabCbrtTab, LabTable::Tab::GAMMA_TAB_SIZE);
                    z = LabTable::Tab::splineInterpolate(vl, __riscv_vfmul(z, LabTable::Tab::GAMMA_TAB_SIZE, vl), LabTable::Tab::instance().LabCbrtTab, LabTable::Tab::GAMMA_TAB_SIZE);

                    lo = __riscv_vmerge(__riscv_vfmul(y, 903.3f, vl), lo, __riscv_vmfgt(y, 0.008856f, vl), vl);
                    ao = __riscv_vfmul(__riscv_vfsub(x, fy, vl), 500.0f, vl);
                    bo = __riscv_vfmul(__riscv_vfsub(fy, z, vl), 200.0f, vl);
                }
                else
                {
                    auto d = __riscv_vfrdiv(__riscv_vfmax(__riscv_vfmadd(y, 15.0f, __riscv_vfmadd(z, 3.0f, x, vl), vl), FLT_EPSILON, vl), 52.0f, vl);
                    ao = __riscv_vfmul(__riscv_vfmadd(x, d, __riscv_vfmv_v_f_f32m2(-2.5719122887f, vl), vl), lo, vl);
                    bo = __riscv_vfmul(__riscv_vfmadd(__riscv_vfmul(y, 2.25f, vl), d, __riscv_vfmv_v_f_f32m2(-6.0884485245f, vl), vl), lo, vl);
                }
            }
            vfloat32m2x3_t vec_dst{};
            vec_dst = __riscv_vset_v_f32m2_f32m2x3(vec_dst, 0, lo);
            vec_dst = __riscv_vset_v_f32m2_f32m2x3(vec_dst, 1, ao);
            vec_dst = __riscv_vset_v_f32m2_f32m2x3(vec_dst, 2, bo);
            __riscv_vsseg3e32(dst + i * dst_step + j * 3, vec_dst, vl);
        }
    }

    return CV_HAL_ERROR_OK;
}

inline int cvtBGRtoLab(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int depth, int scn, bool swapBlue, bool isLab, bool srgb)
{
    if (scn != 3 && scn != 4)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    auto cvtBGRtoLab_b = cvtBGRtoLab_u<true, true>;
    if (!isLab && !srgb)
        cvtBGRtoLab_b = cvtBGRtoLab_u<false, false>;
    else if (!isLab && srgb)
        cvtBGRtoLab_b = cvtBGRtoLab_u<false, true>;
    else if (isLab && !srgb)
        cvtBGRtoLab_b = cvtBGRtoLab_u<true, false>;

    switch (depth)
    {
    case CV_8U:
        return color::invoke(width, height, {cvtBGRtoLab_b}, reinterpret_cast<const uchar*>(src_data), src_step, reinterpret_cast<uchar*>(dst_data), dst_step, width, scn, swapBlue);
    case CV_32F:
        return color::invoke(width, height, {cvtBGRtoLab_f}, reinterpret_cast<const float*>(src_data), src_step, reinterpret_cast<float*>(dst_data), dst_step, width, scn, swapBlue, isLab, srgb);
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}
} // cv::cv_hal_rvv::BGRtoLab

}}

#endif
