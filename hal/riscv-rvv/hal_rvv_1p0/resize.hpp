// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#ifndef OPENCV_HAL_RVV_RESIZE_HPP_INCLUDED
#define OPENCV_HAL_RVV_RESIZE_HPP_INCLUDED

#include <riscv_vector.h>
#include <list>

namespace cv { namespace cv_hal_rvv {

namespace resize {
#undef cv_hal_resize
#define cv_hal_resize cv::cv_hal_rvv::resize::resize

class ResizeInvoker : public ParallelLoopBody
{
public:
    template<typename... Args>
    ResizeInvoker(std::function<int(int, int, Args...)> _func, Args&&... args)
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
static inline int invoke(int height, std::function<int(int, int, Args...)> func, Args&&... args)
{
    cv::parallel_for_(Range(1, height), ResizeInvoker(func, std::forward<Args>(args)...), cv::getNumThreads());
    return func(0, 1, std::forward<Args>(args)...);
}

template<int cn>
static inline int resizeNN(int start, int end, const uchar *src_data, size_t src_step, int src_height, uchar *dst_data, size_t dst_step, int dst_width, int dst_height, double scale_y, int interpolation, const ushort* x_ofs)
{
    const int ify = ((src_height << 16) + dst_height / 2) / dst_height;
    const int ify0 = ify / 2 - src_height % 2;

    for (int i = start; i < end; i++)
    {
        int y_ofs = interpolation == CV_HAL_INTER_NEAREST ? static_cast<int>(std::floor(i * scale_y)) : (ify * i + ify0) >> 16;
        y_ofs = std::min(y_ofs, src_height - 1);

        int vl;
        switch (cn)
        {
        case 1:
            for (int j = 0; j < dst_width; j += vl)
            {
                vl = __riscv_vsetvl_e8m4(dst_width - j);
                auto ptr = __riscv_vle16_v_u16m8(x_ofs + j, vl);
                auto src = __riscv_vloxei16_v_u8m4(src_data + y_ofs * src_step, ptr, vl);
                __riscv_vse8(dst_data + i * dst_step + j, src, vl);
            }
            break;
        case 2:
            for (int j = 0; j < dst_width; j += vl)
            {
                vl = __riscv_vsetvl_e8m4(dst_width - j);
                auto ptr = __riscv_vle16_v_u16m8(x_ofs + j, vl);
                auto src = __riscv_vloxei16_v_u16m8(reinterpret_cast<const ushort*>(src_data + y_ofs * src_step), ptr, vl);
                __riscv_vse16(reinterpret_cast<ushort*>(dst_data + i * dst_step) + j, src, vl);
            }
            break;
        case 3:
            for (int j = 0; j < dst_width; j += vl)
            {
                vl = __riscv_vsetvl_e8m2(dst_width - j);
                auto ptr = __riscv_vle16_v_u16m4(x_ofs + j, vl);
                auto src = __riscv_vloxseg3ei16_v_u8m2x3(src_data + y_ofs * src_step, ptr, vl);
                __riscv_vsseg3e8(dst_data + i * dst_step + j * 3, src, vl);
            }
            break;
        case 4:
            for (int j = 0; j < dst_width; j += vl)
            {
                vl = __riscv_vsetvl_e8m2(dst_width - j);
                auto ptr = __riscv_vle16_v_u16m4(x_ofs + j, vl);
                auto src = __riscv_vloxei16_v_u32m8(reinterpret_cast<const uint*>(src_data + y_ofs * src_step), ptr, vl);
                __riscv_vse32(reinterpret_cast<uint*>(dst_data + i * dst_step) + j, src, vl);
            }
            break;
        default:
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }
    }

    return CV_HAL_ERROR_OK;
}

template<typename helper> struct rvv;
template<> struct rvv<RVV_U8M1>
{
    static inline vfloat32m4_t vcvt0(vuint8m1_t a, size_t b) { return __riscv_vfcvt_f(__riscv_vzext_vf4(a, b), b); }
    static inline vuint8m1_t vcvt1(vfloat32m4_t a, size_t b) { return __riscv_vnclipu(__riscv_vfncvt_xu(a, b), 0, __RISCV_VXRM_RNU, b); }
    static inline vuint8m1_t vloxei(const uchar* a, vuint16m2_t b, size_t c) { return __riscv_vloxei16_v_u8m1(a, b, c); }
    static inline void vloxseg2ei(const uchar* a, vuint16m2_t b, size_t c, vuint8m1_t& x, vuint8m1_t& y) { auto src = __riscv_vloxseg2ei16_v_u8m1x2(a, b, c); x = __riscv_vget_v_u8m1x2_u8m1(src, 0); y = __riscv_vget_v_u8m1x2_u8m1(src, 1); }
    static inline void vloxseg3ei(const uchar* a, vuint16m2_t b, size_t c, vuint8m1_t& x, vuint8m1_t& y, vuint8m1_t& z) { auto src = __riscv_vloxseg3ei16_v_u8m1x3(a, b, c); x = __riscv_vget_v_u8m1x3_u8m1(src, 0); y = __riscv_vget_v_u8m1x3_u8m1(src, 1); z = __riscv_vget_v_u8m1x3_u8m1(src, 2); }
    static inline void vloxseg4ei(const uchar* a, vuint16m2_t b, size_t c, vuint8m1_t& x, vuint8m1_t& y, vuint8m1_t& z, vuint8m1_t& w) { auto src = __riscv_vloxseg4ei16_v_u8m1x4(a, b, c); x = __riscv_vget_v_u8m1x4_u8m1(src, 0); y = __riscv_vget_v_u8m1x4_u8m1(src, 1); z = __riscv_vget_v_u8m1x4_u8m1(src, 2); w = __riscv_vget_v_u8m1x4_u8m1(src, 3); }
    static inline void vsseg2e(uchar* a, size_t b, vuint8m1_t x, vuint8m1_t y) { vuint8m1x2_t dst{}; dst = __riscv_vset_v_u8m1_u8m1x2(dst, 0, x); dst = __riscv_vset_v_u8m1_u8m1x2(dst, 1, y); __riscv_vsseg2e8(a, dst, b); }
    static inline void vsseg3e(uchar* a, size_t b, vuint8m1_t x, vuint8m1_t y, vuint8m1_t z) { vuint8m1x3_t dst{}; dst = __riscv_vset_v_u8m1_u8m1x3(dst, 0, x); dst = __riscv_vset_v_u8m1_u8m1x3(dst, 1, y); dst = __riscv_vset_v_u8m1_u8m1x3(dst, 2, z); __riscv_vsseg3e8(a, dst, b); }
    static inline void vsseg4e(uchar* a, size_t b, vuint8m1_t x, vuint8m1_t y, vuint8m1_t z, vuint8m1_t w) { vuint8m1x4_t dst{}; dst = __riscv_vset_v_u8m1_u8m1x4(dst, 0, x); dst = __riscv_vset_v_u8m1_u8m1x4(dst, 1, y); dst = __riscv_vset_v_u8m1_u8m1x4(dst, 2, z); dst = __riscv_vset_v_u8m1_u8m1x4(dst, 3, w); __riscv_vsseg4e8(a, dst, b); }

    static inline void vlseg2e(const uchar* a, size_t b, vuint8m1_t& x, vuint8m1_t& y) { auto src = __riscv_vlseg2e8_v_u8m1x2(a, b); x = __riscv_vget_v_u8m1x2_u8m1(src, 0); y = __riscv_vget_v_u8m1x2_u8m1(src, 1); }
    static inline void vlsseg2e(const uchar* a, ptrdiff_t b, size_t c, vuint8m1_t& x, vuint8m1_t& y) { auto src = __riscv_vlsseg2e8_v_u8m1x2(a, b, c); x = __riscv_vget_v_u8m1x2_u8m1(src, 0); y = __riscv_vget_v_u8m1x2_u8m1(src, 1); }
    static inline void vlsseg3e(const uchar* a, ptrdiff_t b, size_t c, vuint8m1_t& x, vuint8m1_t& y, vuint8m1_t& z) { auto src = __riscv_vlsseg3e8_v_u8m1x3(a, b, c); x = __riscv_vget_v_u8m1x3_u8m1(src, 0); y = __riscv_vget_v_u8m1x3_u8m1(src, 1); z = __riscv_vget_v_u8m1x3_u8m1(src, 2); }
    static inline void vlsseg4e(const uchar* a, ptrdiff_t b, size_t c, vuint8m1_t& x, vuint8m1_t& y, vuint8m1_t& z, vuint8m1_t& w) { auto src = __riscv_vlsseg4e8_v_u8m1x4(a, b, c); x = __riscv_vget_v_u8m1x4_u8m1(src, 0); y = __riscv_vget_v_u8m1x4_u8m1(src, 1); z = __riscv_vget_v_u8m1x4_u8m1(src, 2); w = __riscv_vget_v_u8m1x4_u8m1(src, 3); }
};
template<> struct rvv<RVV_U16M2>
{
    static inline vfloat32m4_t vcvt0(vuint16m2_t a, size_t b) { return __riscv_vfwcvt_f(a, b); }
    static inline vuint16m2_t vcvt1(vfloat32m4_t a, size_t b) { return __riscv_vfncvt_xu(a, b); }
    static inline vuint16m2_t vloxei(const ushort* a, vuint16m2_t b, size_t c) { return __riscv_vloxei16_v_u16m2(a, b, c); }
    static inline void vloxseg2ei(const ushort* a, vuint16m2_t b, size_t c, vuint16m2_t& x, vuint16m2_t& y) { auto src = __riscv_vloxseg2ei16_v_u16m2x2(a, b, c); x = __riscv_vget_v_u16m2x2_u16m2(src, 0); y = __riscv_vget_v_u16m2x2_u16m2(src, 1); }
    static inline void vloxseg3ei(const ushort* a, vuint16m2_t b, size_t c, vuint16m2_t& x, vuint16m2_t& y, vuint16m2_t& z) { auto src = __riscv_vloxseg3ei16_v_u16m2x3(a, b, c); x = __riscv_vget_v_u16m2x3_u16m2(src, 0); y = __riscv_vget_v_u16m2x3_u16m2(src, 1); z = __riscv_vget_v_u16m2x3_u16m2(src, 2); }
    static inline void vloxseg4ei(const ushort* a, vuint16m2_t b, size_t c, vuint16m2_t& x, vuint16m2_t& y, vuint16m2_t& z, vuint16m2_t& w) { auto src = __riscv_vloxseg4ei16_v_u16m2x4(a, b, c); x = __riscv_vget_v_u16m2x4_u16m2(src, 0); y = __riscv_vget_v_u16m2x4_u16m2(src, 1); z = __riscv_vget_v_u16m2x4_u16m2(src, 2); w = __riscv_vget_v_u16m2x4_u16m2(src, 3); }
    static inline void vsseg2e(ushort* a, size_t b, vuint16m2_t x, vuint16m2_t y) { vuint16m2x2_t dst{}; dst = __riscv_vset_v_u16m2_u16m2x2(dst, 0, x); dst = __riscv_vset_v_u16m2_u16m2x2(dst, 1, y); __riscv_vsseg2e16(a, dst, b); }
    static inline void vsseg3e(ushort* a, size_t b, vuint16m2_t x, vuint16m2_t y, vuint16m2_t z) { vuint16m2x3_t dst{}; dst = __riscv_vset_v_u16m2_u16m2x3(dst, 0, x); dst = __riscv_vset_v_u16m2_u16m2x3(dst, 1, y); dst = __riscv_vset_v_u16m2_u16m2x3(dst, 2, z); __riscv_vsseg3e16(a, dst, b); }
    static inline void vsseg4e(ushort* a, size_t b, vuint16m2_t x, vuint16m2_t y, vuint16m2_t z, vuint16m2_t w) { vuint16m2x4_t dst{}; dst = __riscv_vset_v_u16m2_u16m2x4(dst, 0, x); dst = __riscv_vset_v_u16m2_u16m2x4(dst, 1, y); dst = __riscv_vset_v_u16m2_u16m2x4(dst, 2, z); dst = __riscv_vset_v_u16m2_u16m2x4(dst, 3, w); __riscv_vsseg4e16(a, dst, b); }

    static inline void vlseg2e(const ushort* a, size_t b, vuint16m2_t& x, vuint16m2_t& y) { auto src = __riscv_vlseg2e16_v_u16m2x2(a, b); x = __riscv_vget_v_u16m2x2_u16m2(src, 0); y = __riscv_vget_v_u16m2x2_u16m2(src, 1); }
    static inline void vlsseg2e(const ushort* a, ptrdiff_t b, size_t c, vuint16m2_t& x, vuint16m2_t& y) { auto src = __riscv_vlsseg2e16_v_u16m2x2(a, b, c); x = __riscv_vget_v_u16m2x2_u16m2(src, 0); y = __riscv_vget_v_u16m2x2_u16m2(src, 1); }
    static inline void vlsseg3e(const ushort* a, ptrdiff_t b, size_t c, vuint16m2_t& x, vuint16m2_t& y, vuint16m2_t& z) { auto src = __riscv_vlsseg3e16_v_u16m2x3(a, b, c); x = __riscv_vget_v_u16m2x3_u16m2(src, 0); y = __riscv_vget_v_u16m2x3_u16m2(src, 1); z = __riscv_vget_v_u16m2x3_u16m2(src, 2); }
    static inline void vlsseg4e(const ushort* a, ptrdiff_t b, size_t c, vuint16m2_t& x, vuint16m2_t& y, vuint16m2_t& z, vuint16m2_t& w) { auto src = __riscv_vlsseg4e16_v_u16m2x4(a, b, c); x = __riscv_vget_v_u16m2x4_u16m2(src, 0); y = __riscv_vget_v_u16m2x4_u16m2(src, 1); z = __riscv_vget_v_u16m2x4_u16m2(src, 2); w = __riscv_vget_v_u16m2x4_u16m2(src, 3); }
};
template<> struct rvv<RVV_F32M4>
{
    static inline vfloat32m4_t vcvt0(vfloat32m4_t a, size_t) { return a; }
    static inline vfloat32m4_t vcvt1(vfloat32m4_t a, size_t) { return a; }
    static inline vfloat32m4_t vloxei(const float* a, vuint16m2_t b, size_t c) { return __riscv_vloxei16_v_f32m4(a, b, c); }
    static inline void vloxseg2ei(const float* a, vuint16m2_t b, size_t c, vfloat32m4_t& x, vfloat32m4_t& y) { auto src = __riscv_vloxseg2ei16_v_f32m4x2(a, b, c); x = __riscv_vget_v_f32m4x2_f32m4(src, 0); y = __riscv_vget_v_f32m4x2_f32m4(src, 1); }
    static inline void vloxseg3ei(const float*, vuint16m2_t, size_t, vfloat32m4_t&, vfloat32m4_t&, vfloat32m4_t&) { /*NOTREACHED*/ }
    static inline void vloxseg4ei(const float*, vuint16m2_t, size_t, vfloat32m4_t&, vfloat32m4_t&, vfloat32m4_t&, vfloat32m4_t&) { /*NOTREACHED*/ }
    static inline void vsseg2e(float* a, size_t b, vfloat32m4_t x, vfloat32m4_t y) { vfloat32m4x2_t dst{}; dst = __riscv_vset_v_f32m4_f32m4x2(dst, 0, x); dst = __riscv_vset_v_f32m4_f32m4x2(dst, 1, y); __riscv_vsseg2e32(a, dst, b); }
    static inline void vsseg3e(float*, size_t, vfloat32m4_t, vfloat32m4_t, vfloat32m4_t) { /*NOTREACHED*/ }
    static inline void vsseg4e(float*, size_t, vfloat32m4_t, vfloat32m4_t, vfloat32m4_t, vfloat32m4_t) { /*NOTREACHED*/ }
};
template<> struct rvv<RVV_U8MF2>
{
    static inline vfloat32m2_t vcvt0(vuint8mf2_t a, size_t b) { return __riscv_vfcvt_f(__riscv_vzext_vf4(a, b), b); }
    static inline vuint8mf2_t vcvt1(vfloat32m2_t a, size_t b) { return __riscv_vnclipu(__riscv_vfncvt_xu(a, b), 0, __RISCV_VXRM_RNU, b); }
    static inline vuint8mf2_t vloxei(const uchar* a, vuint16m1_t b, size_t c) { return __riscv_vloxei16_v_u8mf2(a, b, c); }
    static inline void vloxseg2ei(const uchar* a, vuint16m1_t b, size_t c, vuint8mf2_t& x, vuint8mf2_t& y) { auto src = __riscv_vloxseg2ei16_v_u8mf2x2(a, b, c); x = __riscv_vget_v_u8mf2x2_u8mf2(src, 0); y = __riscv_vget_v_u8mf2x2_u8mf2(src, 1); }
    static inline void vloxseg3ei(const uchar* a, vuint16m1_t b, size_t c, vuint8mf2_t& x, vuint8mf2_t& y, vuint8mf2_t& z) { auto src = __riscv_vloxseg3ei16_v_u8mf2x3(a, b, c); x = __riscv_vget_v_u8mf2x3_u8mf2(src, 0); y = __riscv_vget_v_u8mf2x3_u8mf2(src, 1); z = __riscv_vget_v_u8mf2x3_u8mf2(src, 2); }
    static inline void vloxseg4ei(const uchar* a, vuint16m1_t b, size_t c, vuint8mf2_t& x, vuint8mf2_t& y, vuint8mf2_t& z, vuint8mf2_t& w) { auto src = __riscv_vloxseg4ei16_v_u8mf2x4(a, b, c); x = __riscv_vget_v_u8mf2x4_u8mf2(src, 0); y = __riscv_vget_v_u8mf2x4_u8mf2(src, 1); z = __riscv_vget_v_u8mf2x4_u8mf2(src, 2); w = __riscv_vget_v_u8mf2x4_u8mf2(src, 3); }
    static inline void vsseg2e(uchar* a, size_t b, vuint8mf2_t x, vuint8mf2_t y) { vuint8mf2x2_t dst{}; dst = __riscv_vset_v_u8mf2_u8mf2x2(dst, 0, x); dst = __riscv_vset_v_u8mf2_u8mf2x2(dst, 1, y); __riscv_vsseg2e8(a, dst, b); }
    static inline void vsseg3e(uchar* a, size_t b, vuint8mf2_t x, vuint8mf2_t y, vuint8mf2_t z) { vuint8mf2x3_t dst{}; dst = __riscv_vset_v_u8mf2_u8mf2x3(dst, 0, x); dst = __riscv_vset_v_u8mf2_u8mf2x3(dst, 1, y); dst = __riscv_vset_v_u8mf2_u8mf2x3(dst, 2, z); __riscv_vsseg3e8(a, dst, b); }
    static inline void vsseg4e(uchar* a, size_t b, vuint8mf2_t x, vuint8mf2_t y, vuint8mf2_t z, vuint8mf2_t w) { vuint8mf2x4_t dst{}; dst = __riscv_vset_v_u8mf2_u8mf2x4(dst, 0, x); dst = __riscv_vset_v_u8mf2_u8mf2x4(dst, 1, y); dst = __riscv_vset_v_u8mf2_u8mf2x4(dst, 2, z); dst = __riscv_vset_v_u8mf2_u8mf2x4(dst, 3, w); __riscv_vsseg4e8(a, dst, b); }
};
template<> struct rvv<RVV_U16M1>
{
    static inline vfloat32m2_t vcvt0(vuint16m1_t a, size_t b) { return __riscv_vfwcvt_f(a, b); }
    static inline vuint16m1_t vcvt1(vfloat32m2_t a, size_t b) { return __riscv_vfncvt_xu(a, b); }
    static inline vuint16m1_t vloxei(const ushort* a, vuint16m1_t b, size_t c) { return __riscv_vloxei16_v_u16m1(a, b, c); }
    static inline void vloxseg2ei(const ushort* a, vuint16m1_t b, size_t c, vuint16m1_t& x, vuint16m1_t& y) { auto src = __riscv_vloxseg2ei16_v_u16m1x2(a, b, c); x = __riscv_vget_v_u16m1x2_u16m1(src, 0); y = __riscv_vget_v_u16m1x2_u16m1(src, 1); }
    static inline void vloxseg3ei(const ushort* a, vuint16m1_t b, size_t c, vuint16m1_t& x, vuint16m1_t& y, vuint16m1_t& z) { auto src = __riscv_vloxseg3ei16_v_u16m1x3(a, b, c); x = __riscv_vget_v_u16m1x3_u16m1(src, 0); y = __riscv_vget_v_u16m1x3_u16m1(src, 1); z = __riscv_vget_v_u16m1x3_u16m1(src, 2); }
    static inline void vloxseg4ei(const ushort* a, vuint16m1_t b, size_t c, vuint16m1_t& x, vuint16m1_t& y, vuint16m1_t& z, vuint16m1_t& w) { auto src = __riscv_vloxseg4ei16_v_u16m1x4(a, b, c); x = __riscv_vget_v_u16m1x4_u16m1(src, 0); y = __riscv_vget_v_u16m1x4_u16m1(src, 1); z = __riscv_vget_v_u16m1x4_u16m1(src, 2); w = __riscv_vget_v_u16m1x4_u16m1(src, 3); }
    static inline void vsseg2e(ushort* a, size_t b, vuint16m1_t x, vuint16m1_t y) { vuint16m1x2_t dst{}; dst = __riscv_vset_v_u16m1_u16m1x2(dst, 0, x); dst = __riscv_vset_v_u16m1_u16m1x2(dst, 1, y); __riscv_vsseg2e16(a, dst, b); }
    static inline void vsseg3e(ushort* a, size_t b, vuint16m1_t x, vuint16m1_t y, vuint16m1_t z) { vuint16m1x3_t dst{}; dst = __riscv_vset_v_u16m1_u16m1x3(dst, 0, x); dst = __riscv_vset_v_u16m1_u16m1x3(dst, 1, y); dst = __riscv_vset_v_u16m1_u16m1x3(dst, 2, z); __riscv_vsseg3e16(a, dst, b); }
    static inline void vsseg4e(ushort* a, size_t b, vuint16m1_t x, vuint16m1_t y, vuint16m1_t z, vuint16m1_t w) { vuint16m1x4_t dst{}; dst = __riscv_vset_v_u16m1_u16m1x4(dst, 0, x); dst = __riscv_vset_v_u16m1_u16m1x4(dst, 1, y); dst = __riscv_vset_v_u16m1_u16m1x4(dst, 2, z); dst = __riscv_vset_v_u16m1_u16m1x4(dst, 3, w); __riscv_vsseg4e16(a, dst, b); }
};
template<> struct rvv<RVV_F32M2>
{
    static inline vfloat32m2_t vcvt0(vfloat32m2_t a, size_t) { return a; }
    static inline vfloat32m2_t vcvt1(vfloat32m2_t a, size_t) { return a; }
    static inline vfloat32m2_t vloxei(const float* a, vuint16m1_t b, size_t c) { return __riscv_vloxei16_v_f32m2(a, b, c); }
    static inline void vloxseg2ei(const float* a, vuint16m1_t b, size_t c, vfloat32m2_t& x, vfloat32m2_t& y) { auto src = __riscv_vloxseg2ei16_v_f32m2x2(a, b, c); x = __riscv_vget_v_f32m2x2_f32m2(src, 0); y = __riscv_vget_v_f32m2x2_f32m2(src, 1); }
    static inline void vloxseg3ei(const float* a, vuint16m1_t b, size_t c, vfloat32m2_t& x, vfloat32m2_t& y, vfloat32m2_t& z) { auto src = __riscv_vloxseg3ei16_v_f32m2x3(a, b, c); x = __riscv_vget_v_f32m2x3_f32m2(src, 0); y = __riscv_vget_v_f32m2x3_f32m2(src, 1); z = __riscv_vget_v_f32m2x3_f32m2(src, 2); }
    static inline void vloxseg4ei(const float* a, vuint16m1_t b, size_t c, vfloat32m2_t& x, vfloat32m2_t& y, vfloat32m2_t& z, vfloat32m2_t& w) { auto src = __riscv_vloxseg4ei16_v_f32m2x4(a, b, c); x = __riscv_vget_v_f32m2x4_f32m2(src, 0); y = __riscv_vget_v_f32m2x4_f32m2(src, 1); z = __riscv_vget_v_f32m2x4_f32m2(src, 2); w = __riscv_vget_v_f32m2x4_f32m2(src, 3); }
    static inline void vsseg2e(float* a, size_t b, vfloat32m2_t x, vfloat32m2_t y) { vfloat32m2x2_t dst{}; dst = __riscv_vset_v_f32m2_f32m2x2(dst, 0, x); dst = __riscv_vset_v_f32m2_f32m2x2(dst, 1, y); __riscv_vsseg2e32(a, dst, b); }
    static inline void vsseg3e(float* a, size_t b, vfloat32m2_t x, vfloat32m2_t y, vfloat32m2_t z) { vfloat32m2x3_t dst{}; dst = __riscv_vset_v_f32m2_f32m2x3(dst, 0, x); dst = __riscv_vset_v_f32m2_f32m2x3(dst, 1, y); dst = __riscv_vset_v_f32m2_f32m2x3(dst, 2, z); __riscv_vsseg3e32(a, dst, b); }
    static inline void vsseg4e(float* a, size_t b, vfloat32m2_t x, vfloat32m2_t y, vfloat32m2_t z, vfloat32m2_t w) { vfloat32m2x4_t dst{}; dst = __riscv_vset_v_f32m2_f32m2x4(dst, 0, x); dst = __riscv_vset_v_f32m2_f32m2x4(dst, 1, y); dst = __riscv_vset_v_f32m2_f32m2x4(dst, 2, z); dst = __riscv_vset_v_f32m2_f32m2x4(dst, 3, w); __riscv_vsseg4e32(a, dst, b); }
};

template<typename helper, int cn>
static inline int resizeLinear(int start, int end, const uchar *src_data, size_t src_step, int src_height, uchar *dst_data, size_t dst_step, int dst_width, double scale_y, const ushort* x_ofs0, const ushort* x_ofs1, const float* x_val)
{
    using T = typename helper::ElemType;

    for (int i = start; i < end; i++)
    {
        float my = (i + 0.5) * scale_y - 0.5;
        int y_ofs = static_cast<int>(std::floor(my));
        my -= y_ofs;

        int y_ofs0 = std::min(std::max(y_ofs    , 0), src_height - 1);
        int y_ofs1 = std::min(std::max(y_ofs + 1, 0), src_height - 1);

        int vl;
        switch (cn)
        {
        case 1:
            for (int j = 0; j < dst_width; j += vl)
            {
                vl = helper::setvl(dst_width - j);
                auto ptr0 = RVV_SameLen<ushort, helper>::vload(x_ofs0 + j, vl);
                auto ptr1 = RVV_SameLen<ushort, helper>::vload(x_ofs1 + j, vl);

                auto v0 = rvv<helper>::vcvt0(rvv<helper>::vloxei(reinterpret_cast<const T*>(src_data + y_ofs0 * src_step), ptr0, vl), vl);
                auto v1 = rvv<helper>::vcvt0(rvv<helper>::vloxei(reinterpret_cast<const T*>(src_data + y_ofs0 * src_step), ptr1, vl), vl);
                auto v2 = rvv<helper>::vcvt0(rvv<helper>::vloxei(reinterpret_cast<const T*>(src_data + y_ofs1 * src_step), ptr0, vl), vl);
                auto v3 = rvv<helper>::vcvt0(rvv<helper>::vloxei(reinterpret_cast<const T*>(src_data + y_ofs1 * src_step), ptr1, vl), vl);

                auto mx = RVV_SameLen<float, helper>::vload(x_val + j, vl);
                v0 = __riscv_vfmadd(__riscv_vfsub(v1, v0, vl), mx, v0, vl);
                v2 = __riscv_vfmadd(__riscv_vfsub(v3, v2, vl), mx, v2, vl);
                v0 = __riscv_vfmadd(__riscv_vfsub(v2, v0, vl), my, v0, vl);
                helper::vstore(reinterpret_cast<T*>(dst_data + i * dst_step) + j, rvv<helper>::vcvt1(v0, vl), vl);
            }
            break;
        case 2:
            for (int j = 0; j < dst_width; j += vl)
            {
                vl = helper::setvl(dst_width - j);
                auto ptr0 = RVV_SameLen<ushort, helper>::vload(x_ofs0 + j, vl);
                auto ptr1 = RVV_SameLen<ushort, helper>::vload(x_ofs1 + j, vl);

                typename helper::VecType s0, s1;
                rvv<helper>::vloxseg2ei(reinterpret_cast<const T*>(src_data + y_ofs0 * src_step), ptr0, vl, s0, s1);
                auto v00 = rvv<helper>::vcvt0(s0, vl), v10 = rvv<helper>::vcvt0(s1, vl);
                rvv<helper>::vloxseg2ei(reinterpret_cast<const T*>(src_data + y_ofs0 * src_step), ptr1, vl, s0, s1);
                auto v01 = rvv<helper>::vcvt0(s0, vl), v11 = rvv<helper>::vcvt0(s1, vl);
                rvv<helper>::vloxseg2ei(reinterpret_cast<const T*>(src_data + y_ofs1 * src_step), ptr0, vl, s0, s1);
                auto v02 = rvv<helper>::vcvt0(s0, vl), v12 = rvv<helper>::vcvt0(s1, vl);
                rvv<helper>::vloxseg2ei(reinterpret_cast<const T*>(src_data + y_ofs1 * src_step), ptr1, vl, s0, s1);
                auto v03 = rvv<helper>::vcvt0(s0, vl), v13 = rvv<helper>::vcvt0(s1, vl);

                auto mx = RVV_SameLen<float, helper>::vload(x_val + j, vl);
                v00 = __riscv_vfmadd(__riscv_vfsub(v01, v00, vl), mx, v00, vl);
                v02 = __riscv_vfmadd(__riscv_vfsub(v03, v02, vl), mx, v02, vl);
                v00 = __riscv_vfmadd(__riscv_vfsub(v02, v00, vl), my, v00, vl);
                v10 = __riscv_vfmadd(__riscv_vfsub(v11, v10, vl), mx, v10, vl);
                v12 = __riscv_vfmadd(__riscv_vfsub(v13, v12, vl), mx, v12, vl);
                v10 = __riscv_vfmadd(__riscv_vfsub(v12, v10, vl), my, v10, vl);
                rvv<helper>::vsseg2e(reinterpret_cast<T*>(dst_data + i * dst_step) + j * 2, vl, rvv<helper>::vcvt1(v00, vl), rvv<helper>::vcvt1(v10, vl));
            }
            break;
        case 3:
            for (int j = 0; j < dst_width; j += vl)
            {
                vl = helper::setvl(dst_width - j);
                auto ptr0 = RVV_SameLen<ushort, helper>::vload(x_ofs0 + j, vl);
                auto ptr1 = RVV_SameLen<ushort, helper>::vload(x_ofs1 + j, vl);

                typename helper::VecType s0, s1, s2;
                rvv<helper>::vloxseg3ei(reinterpret_cast<const T*>(src_data + y_ofs0 * src_step), ptr0, vl, s0, s1, s2);
                auto v00 = rvv<helper>::vcvt0(s0, vl), v10 = rvv<helper>::vcvt0(s1, vl), v20 = rvv<helper>::vcvt0(s2, vl);
                rvv<helper>::vloxseg3ei(reinterpret_cast<const T*>(src_data + y_ofs0 * src_step), ptr1, vl, s0, s1, s2);
                auto v01 = rvv<helper>::vcvt0(s0, vl), v11 = rvv<helper>::vcvt0(s1, vl), v21 = rvv<helper>::vcvt0(s2, vl);
                rvv<helper>::vloxseg3ei(reinterpret_cast<const T*>(src_data + y_ofs1 * src_step), ptr0, vl, s0, s1, s2);
                auto v02 = rvv<helper>::vcvt0(s0, vl), v12 = rvv<helper>::vcvt0(s1, vl), v22 = rvv<helper>::vcvt0(s2, vl);
                rvv<helper>::vloxseg3ei(reinterpret_cast<const T*>(src_data + y_ofs1 * src_step), ptr1, vl, s0, s1, s2);
                auto v03 = rvv<helper>::vcvt0(s0, vl), v13 = rvv<helper>::vcvt0(s1, vl), v23 = rvv<helper>::vcvt0(s2, vl);

                auto mx = RVV_SameLen<float, helper>::vload(x_val + j, vl);
                v00 = __riscv_vfmadd(__riscv_vfsub(v01, v00, vl), mx, v00, vl);
                v02 = __riscv_vfmadd(__riscv_vfsub(v03, v02, vl), mx, v02, vl);
                v00 = __riscv_vfmadd(__riscv_vfsub(v02, v00, vl), my, v00, vl);
                v10 = __riscv_vfmadd(__riscv_vfsub(v11, v10, vl), mx, v10, vl);
                v12 = __riscv_vfmadd(__riscv_vfsub(v13, v12, vl), mx, v12, vl);
                v10 = __riscv_vfmadd(__riscv_vfsub(v12, v10, vl), my, v10, vl);
                v20 = __riscv_vfmadd(__riscv_vfsub(v21, v20, vl), mx, v20, vl);
                v22 = __riscv_vfmadd(__riscv_vfsub(v23, v22, vl), mx, v22, vl);
                v20 = __riscv_vfmadd(__riscv_vfsub(v22, v20, vl), my, v20, vl);
                rvv<helper>::vsseg3e(reinterpret_cast<T*>(dst_data + i * dst_step) + j * 3, vl, rvv<helper>::vcvt1(v00, vl), rvv<helper>::vcvt1(v10, vl), rvv<helper>::vcvt1(v20, vl));
            }
            break;
        case 4:
            for (int j = 0; j < dst_width; j += vl)
            {
                vl = helper::setvl(dst_width - j);
                auto ptr0 = RVV_SameLen<ushort, helper>::vload(x_ofs0 + j, vl);
                auto ptr1 = RVV_SameLen<ushort, helper>::vload(x_ofs1 + j, vl);

                typename helper::VecType s0, s1, s2, s3;
                rvv<helper>::vloxseg4ei(reinterpret_cast<const T*>(src_data + y_ofs0 * src_step), ptr0, vl, s0, s1, s2, s3);
                auto v00 = rvv<helper>::vcvt0(s0, vl), v10 = rvv<helper>::vcvt0(s1, vl), v20 = rvv<helper>::vcvt0(s2, vl), v30 = rvv<helper>::vcvt0(s3, vl);
                rvv<helper>::vloxseg4ei(reinterpret_cast<const T*>(src_data + y_ofs0 * src_step), ptr1, vl, s0, s1, s2, s3);
                auto v01 = rvv<helper>::vcvt0(s0, vl), v11 = rvv<helper>::vcvt0(s1, vl), v21 = rvv<helper>::vcvt0(s2, vl), v31 = rvv<helper>::vcvt0(s3, vl);
                rvv<helper>::vloxseg4ei(reinterpret_cast<const T*>(src_data + y_ofs1 * src_step), ptr0, vl, s0, s1, s2, s3);
                auto v02 = rvv<helper>::vcvt0(s0, vl), v12 = rvv<helper>::vcvt0(s1, vl), v22 = rvv<helper>::vcvt0(s2, vl), v32 = rvv<helper>::vcvt0(s3, vl);
                rvv<helper>::vloxseg4ei(reinterpret_cast<const T*>(src_data + y_ofs1 * src_step), ptr1, vl, s0, s1, s2, s3);
                auto v03 = rvv<helper>::vcvt0(s0, vl), v13 = rvv<helper>::vcvt0(s1, vl), v23 = rvv<helper>::vcvt0(s2, vl), v33 = rvv<helper>::vcvt0(s3, vl);

                auto mx = RVV_SameLen<float, helper>::vload(x_val + j, vl);
                v00 = __riscv_vfmadd(__riscv_vfsub(v01, v00, vl), mx, v00, vl);
                v02 = __riscv_vfmadd(__riscv_vfsub(v03, v02, vl), mx, v02, vl);
                v00 = __riscv_vfmadd(__riscv_vfsub(v02, v00, vl), my, v00, vl);
                v10 = __riscv_vfmadd(__riscv_vfsub(v11, v10, vl), mx, v10, vl);
                v12 = __riscv_vfmadd(__riscv_vfsub(v13, v12, vl), mx, v12, vl);
                v10 = __riscv_vfmadd(__riscv_vfsub(v12, v10, vl), my, v10, vl);
                v20 = __riscv_vfmadd(__riscv_vfsub(v21, v20, vl), mx, v20, vl);
                v22 = __riscv_vfmadd(__riscv_vfsub(v23, v22, vl), mx, v22, vl);
                v20 = __riscv_vfmadd(__riscv_vfsub(v22, v20, vl), my, v20, vl);
                v30 = __riscv_vfmadd(__riscv_vfsub(v31, v30, vl), mx, v30, vl);
                v32 = __riscv_vfmadd(__riscv_vfsub(v33, v32, vl), mx, v32, vl);
                v30 = __riscv_vfmadd(__riscv_vfsub(v32, v30, vl), my, v30, vl);
                rvv<helper>::vsseg4e(reinterpret_cast<T*>(dst_data + i * dst_step) + j * 4, vl, rvv<helper>::vcvt1(v00, vl), rvv<helper>::vcvt1(v10, vl), rvv<helper>::vcvt1(v20, vl), rvv<helper>::vcvt1(v30, vl));
            }
            break;
        default:
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }
    }

    return CV_HAL_ERROR_OK;
}

template<int cn>
static inline int resizeLinearExact(int start, int end, const uchar *src_data, size_t src_step, int src_height, uchar *dst_data, size_t dst_step, int dst_width, double scale_y, const ushort* x_ofs0, const ushort* x_ofs1, const ushort* x_val)
{
    for (int i = start; i < end; i++)
    {
        double y_val = (i + 0.5) * scale_y - 0.5;
        int y_ofs = static_cast<int>(std::floor(y_val));
        y_val -= y_ofs;

        int y_ofs0 = std::min(std::max(y_ofs    , 0), src_height - 1);
        int y_ofs1 = std::min(std::max(y_ofs + 1, 0), src_height - 1);
        ushort my = static_cast<ushort>(y_val * 256 - std::remainder(y_val * 256, 1));

        int vl;
        switch (cn)
        {
        case 1:
            for (int j = 0; j < dst_width; j += vl)
            {
                vl = __riscv_vsetvl_e8m1(dst_width - j);
                auto ptr0 = __riscv_vle16_v_u16m2(x_ofs0 + j, vl);
                auto ptr1 = __riscv_vle16_v_u16m2(x_ofs1 + j, vl);

                auto v0 = __riscv_vzext_vf2(__riscv_vloxei16_v_u8m1(src_data + y_ofs0 * src_step, ptr0, vl), vl);
                auto v1 = __riscv_vzext_vf2(__riscv_vloxei16_v_u8m1(src_data + y_ofs0 * src_step, ptr1, vl), vl);
                auto v2 = __riscv_vzext_vf2(__riscv_vloxei16_v_u8m1(src_data + y_ofs1 * src_step, ptr0, vl), vl);
                auto v3 = __riscv_vzext_vf2(__riscv_vloxei16_v_u8m1(src_data + y_ofs1 * src_step, ptr1, vl), vl);

                auto mx = __riscv_vle16_v_u16m2(x_val + j, vl);
                v0 = __riscv_vmadd(v0, __riscv_vrsub(mx, 256, vl), __riscv_vmul(v1, mx, vl), vl);
                v2 = __riscv_vmadd(v2, __riscv_vrsub(mx, 256, vl), __riscv_vmul(v3, mx, vl), vl);
                auto d0 = __riscv_vwmaccu(__riscv_vwmulu(v2, my, vl), 256 - my, v0, vl);
                __riscv_vse8(dst_data + i * dst_step + j, __riscv_vnclipu(__riscv_vnclipu(d0, 16, __RISCV_VXRM_RNU, vl), 0, __RISCV_VXRM_RNU, vl), vl);
            }
            break;
        case 2:
            for (int j = 0; j < dst_width; j += vl)
            {
                vl = __riscv_vsetvl_e8m1(dst_width - j);
                auto ptr0 = __riscv_vle16_v_u16m2(x_ofs0 + j, vl);
                auto ptr1 = __riscv_vle16_v_u16m2(x_ofs1 + j, vl);

                auto src = __riscv_vloxseg2ei16_v_u8m1x2(src_data + y_ofs0 * src_step, ptr0, vl);
                auto v00 = __riscv_vzext_vf2(__riscv_vget_v_u8m1x2_u8m1(src, 0), vl), v10 = __riscv_vzext_vf2(__riscv_vget_v_u8m1x2_u8m1(src, 1), vl);
                src = __riscv_vloxseg2ei16_v_u8m1x2(src_data + y_ofs0 * src_step, ptr1, vl);
                auto v01 = __riscv_vzext_vf2(__riscv_vget_v_u8m1x2_u8m1(src, 0), vl), v11 = __riscv_vzext_vf2(__riscv_vget_v_u8m1x2_u8m1(src, 1), vl);
                src = __riscv_vloxseg2ei16_v_u8m1x2(src_data + y_ofs1 * src_step, ptr0, vl);
                auto v02 = __riscv_vzext_vf2(__riscv_vget_v_u8m1x2_u8m1(src, 0), vl), v12 = __riscv_vzext_vf2(__riscv_vget_v_u8m1x2_u8m1(src, 1), vl);
                src = __riscv_vloxseg2ei16_v_u8m1x2(src_data + y_ofs1 * src_step, ptr1, vl);
                auto v03 = __riscv_vzext_vf2(__riscv_vget_v_u8m1x2_u8m1(src, 0), vl), v13 = __riscv_vzext_vf2(__riscv_vget_v_u8m1x2_u8m1(src, 1), vl);

                auto mx = __riscv_vle16_v_u16m2(x_val + j, vl);
                v00 = __riscv_vmadd(v00, __riscv_vrsub(mx, 256, vl), __riscv_vmul(v01, mx, vl), vl);
                v02 = __riscv_vmadd(v02, __riscv_vrsub(mx, 256, vl), __riscv_vmul(v03, mx, vl), vl);
                auto d00 = __riscv_vwmaccu(__riscv_vwmulu(v02, my, vl), 256 - my, v00, vl);
                v10 = __riscv_vmadd(v10, __riscv_vrsub(mx, 256, vl), __riscv_vmul(v11, mx, vl), vl);
                v12 = __riscv_vmadd(v12, __riscv_vrsub(mx, 256, vl), __riscv_vmul(v13, mx, vl), vl);
                auto d10 = __riscv_vwmaccu(__riscv_vwmulu(v12, my, vl), 256 - my, v10, vl);

                vuint8m1x2_t dst{};
                dst = __riscv_vset_v_u8m1_u8m1x2(dst, 0, __riscv_vnclipu(__riscv_vnclipu(d00, 16, __RISCV_VXRM_RNU, vl), 0, __RISCV_VXRM_RNU, vl));
                dst = __riscv_vset_v_u8m1_u8m1x2(dst, 1, __riscv_vnclipu(__riscv_vnclipu(d10, 16, __RISCV_VXRM_RNU, vl), 0, __RISCV_VXRM_RNU, vl));
                __riscv_vsseg2e8(dst_data + i * dst_step + j * 2, dst, vl);
            }
            break;
        case 3:
            for (int j = 0; j < dst_width; j += vl)
            {
                vl = __riscv_vsetvl_e8mf2(dst_width - j);
                auto ptr0 = __riscv_vle16_v_u16m1(x_ofs0 + j, vl);
                auto ptr1 = __riscv_vle16_v_u16m1(x_ofs1 + j, vl);

                auto src = __riscv_vloxseg3ei16_v_u8mf2x3(src_data + y_ofs0 * src_step, ptr0, vl);
                auto v00 = __riscv_vzext_vf2(__riscv_vget_v_u8mf2x3_u8mf2(src, 0), vl), v10 = __riscv_vzext_vf2(__riscv_vget_v_u8mf2x3_u8mf2(src, 1), vl), v20 = __riscv_vzext_vf2(__riscv_vget_v_u8mf2x3_u8mf2(src, 2), vl);
                src = __riscv_vloxseg3ei16_v_u8mf2x3(src_data + y_ofs0 * src_step, ptr1, vl);
                auto v01 = __riscv_vzext_vf2(__riscv_vget_v_u8mf2x3_u8mf2(src, 0), vl), v11 = __riscv_vzext_vf2(__riscv_vget_v_u8mf2x3_u8mf2(src, 1), vl), v21 = __riscv_vzext_vf2(__riscv_vget_v_u8mf2x3_u8mf2(src, 2), vl);
                src = __riscv_vloxseg3ei16_v_u8mf2x3(src_data + y_ofs1 * src_step, ptr0, vl);
                auto v02 = __riscv_vzext_vf2(__riscv_vget_v_u8mf2x3_u8mf2(src, 0), vl), v12 = __riscv_vzext_vf2(__riscv_vget_v_u8mf2x3_u8mf2(src, 1), vl), v22 = __riscv_vzext_vf2(__riscv_vget_v_u8mf2x3_u8mf2(src, 2), vl);
                src = __riscv_vloxseg3ei16_v_u8mf2x3(src_data + y_ofs1 * src_step, ptr1, vl);
                auto v03 = __riscv_vzext_vf2(__riscv_vget_v_u8mf2x3_u8mf2(src, 0), vl), v13 = __riscv_vzext_vf2(__riscv_vget_v_u8mf2x3_u8mf2(src, 1), vl), v23 = __riscv_vzext_vf2(__riscv_vget_v_u8mf2x3_u8mf2(src, 2), vl);

                auto mx = __riscv_vle16_v_u16m1(x_val + j, vl);
                v00 = __riscv_vmadd(v00, __riscv_vrsub(mx, 256, vl), __riscv_vmul(v01, mx, vl), vl);
                v02 = __riscv_vmadd(v02, __riscv_vrsub(mx, 256, vl), __riscv_vmul(v03, mx, vl), vl);
                auto d00 = __riscv_vwmaccu(__riscv_vwmulu(v02, my, vl), 256 - my, v00, vl);
                v10 = __riscv_vmadd(v10, __riscv_vrsub(mx, 256, vl), __riscv_vmul(v11, mx, vl), vl);
                v12 = __riscv_vmadd(v12, __riscv_vrsub(mx, 256, vl), __riscv_vmul(v13, mx, vl), vl);
                auto d10 = __riscv_vwmaccu(__riscv_vwmulu(v12, my, vl), 256 - my, v10, vl);
                v20 = __riscv_vmadd(v20, __riscv_vrsub(mx, 256, vl), __riscv_vmul(v21, mx, vl), vl);
                v22 = __riscv_vmadd(v22, __riscv_vrsub(mx, 256, vl), __riscv_vmul(v23, mx, vl), vl);
                auto d20 = __riscv_vwmaccu(__riscv_vwmulu(v22, my, vl), 256 - my, v20, vl);

                vuint8mf2x3_t dst{};
                dst = __riscv_vset_v_u8mf2_u8mf2x3(dst, 0, __riscv_vnclipu(__riscv_vnclipu(d00, 16, __RISCV_VXRM_RNU, vl), 0, __RISCV_VXRM_RNU, vl));
                dst = __riscv_vset_v_u8mf2_u8mf2x3(dst, 1, __riscv_vnclipu(__riscv_vnclipu(d10, 16, __RISCV_VXRM_RNU, vl), 0, __RISCV_VXRM_RNU, vl));
                dst = __riscv_vset_v_u8mf2_u8mf2x3(dst, 2, __riscv_vnclipu(__riscv_vnclipu(d20, 16, __RISCV_VXRM_RNU, vl), 0, __RISCV_VXRM_RNU, vl));
                __riscv_vsseg3e8(dst_data + i * dst_step + j * 3, dst, vl);
            }
            break;
        case 4:
            for (int j = 0; j < dst_width; j += vl)
            {
                vl = __riscv_vsetvl_e8mf2(dst_width - j);
                auto ptr0 = __riscv_vle16_v_u16m1(x_ofs0 + j, vl);
                auto ptr1 = __riscv_vle16_v_u16m1(x_ofs1 + j, vl);

                auto src = __riscv_vloxseg4ei16_v_u8mf2x4(src_data + y_ofs0 * src_step, ptr0, vl);
                auto v00 = __riscv_vzext_vf2(__riscv_vget_v_u8mf2x4_u8mf2(src, 0), vl), v10 = __riscv_vzext_vf2(__riscv_vget_v_u8mf2x4_u8mf2(src, 1), vl), v20 = __riscv_vzext_vf2(__riscv_vget_v_u8mf2x4_u8mf2(src, 2), vl), v30 = __riscv_vzext_vf2(__riscv_vget_v_u8mf2x4_u8mf2(src, 3), vl);
                src = __riscv_vloxseg4ei16_v_u8mf2x4(src_data + y_ofs0 * src_step, ptr1, vl);
                auto v01 = __riscv_vzext_vf2(__riscv_vget_v_u8mf2x4_u8mf2(src, 0), vl), v11 = __riscv_vzext_vf2(__riscv_vget_v_u8mf2x4_u8mf2(src, 1), vl), v21 = __riscv_vzext_vf2(__riscv_vget_v_u8mf2x4_u8mf2(src, 2), vl), v31 = __riscv_vzext_vf2(__riscv_vget_v_u8mf2x4_u8mf2(src, 3), vl);
                src = __riscv_vloxseg4ei16_v_u8mf2x4(src_data + y_ofs1 * src_step, ptr0, vl);
                auto v02 = __riscv_vzext_vf2(__riscv_vget_v_u8mf2x4_u8mf2(src, 0), vl), v12 = __riscv_vzext_vf2(__riscv_vget_v_u8mf2x4_u8mf2(src, 1), vl), v22 = __riscv_vzext_vf2(__riscv_vget_v_u8mf2x4_u8mf2(src, 2), vl), v32 = __riscv_vzext_vf2(__riscv_vget_v_u8mf2x4_u8mf2(src, 3), vl);
                src = __riscv_vloxseg4ei16_v_u8mf2x4(src_data + y_ofs1 * src_step, ptr1, vl);
                auto v03 = __riscv_vzext_vf2(__riscv_vget_v_u8mf2x4_u8mf2(src, 0), vl), v13 = __riscv_vzext_vf2(__riscv_vget_v_u8mf2x4_u8mf2(src, 1), vl), v23 = __riscv_vzext_vf2(__riscv_vget_v_u8mf2x4_u8mf2(src, 2), vl), v33 = __riscv_vzext_vf2(__riscv_vget_v_u8mf2x4_u8mf2(src, 3), vl);

                auto mx = __riscv_vle16_v_u16m1(x_val + j, vl);
                v00 = __riscv_vmadd(v00, __riscv_vrsub(mx, 256, vl), __riscv_vmul(v01, mx, vl), vl);
                v02 = __riscv_vmadd(v02, __riscv_vrsub(mx, 256, vl), __riscv_vmul(v03, mx, vl), vl);
                auto d00 = __riscv_vwmaccu(__riscv_vwmulu(v02, my, vl), 256 - my, v00, vl);
                v10 = __riscv_vmadd(v10, __riscv_vrsub(mx, 256, vl), __riscv_vmul(v11, mx, vl), vl);
                v12 = __riscv_vmadd(v12, __riscv_vrsub(mx, 256, vl), __riscv_vmul(v13, mx, vl), vl);
                auto d10 = __riscv_vwmaccu(__riscv_vwmulu(v12, my, vl), 256 - my, v10, vl);
                v20 = __riscv_vmadd(v20, __riscv_vrsub(mx, 256, vl), __riscv_vmul(v21, mx, vl), vl);
                v22 = __riscv_vmadd(v22, __riscv_vrsub(mx, 256, vl), __riscv_vmul(v23, mx, vl), vl);
                auto d20 = __riscv_vwmaccu(__riscv_vwmulu(v22, my, vl), 256 - my, v20, vl);
                v30 = __riscv_vmadd(v30, __riscv_vrsub(mx, 256, vl), __riscv_vmul(v31, mx, vl), vl);
                v32 = __riscv_vmadd(v32, __riscv_vrsub(mx, 256, vl), __riscv_vmul(v33, mx, vl), vl);
                auto d30 = __riscv_vwmaccu(__riscv_vwmulu(v32, my, vl), 256 - my, v30, vl);

                vuint8mf2x4_t dst{};
                dst = __riscv_vset_v_u8mf2_u8mf2x4(dst, 0, __riscv_vnclipu(__riscv_vnclipu(d00, 16, __RISCV_VXRM_RNU, vl), 0, __RISCV_VXRM_RNU, vl));
                dst = __riscv_vset_v_u8mf2_u8mf2x4(dst, 1, __riscv_vnclipu(__riscv_vnclipu(d10, 16, __RISCV_VXRM_RNU, vl), 0, __RISCV_VXRM_RNU, vl));
                dst = __riscv_vset_v_u8mf2_u8mf2x4(dst, 2, __riscv_vnclipu(__riscv_vnclipu(d20, 16, __RISCV_VXRM_RNU, vl), 0, __RISCV_VXRM_RNU, vl));
                dst = __riscv_vset_v_u8mf2_u8mf2x4(dst, 3, __riscv_vnclipu(__riscv_vnclipu(d30, 16, __RISCV_VXRM_RNU, vl), 0, __RISCV_VXRM_RNU, vl));
                __riscv_vsseg4e8(dst_data + i * dst_step + j * 4, dst, vl);
            }
            break;
        default:
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }
    }

    return CV_HAL_ERROR_OK;
}

static int computeResizeAreaTab(int ssize, int dsize, double scale, ushort* stab, ushort* dtab, float* vtab)
{
    int k = 0;
    for (int dx = 0; dx < dsize; dx++)
    {
        double fsx1 = dx * scale;
        double fsx2 = fsx1 + scale;
        double cellWidth = std::min(scale, ssize - fsx1);

        int sx1 = std::ceil(fsx1), sx2 = std::floor(fsx2);

        sx2 = std::min(sx2, ssize - 1);
        sx1 = std::min(sx1, sx2);

        if (sx1 - fsx1 > 1e-3)
        {
            dtab[k] = dx;
            stab[k] = sx1 - 1;
            vtab[k++] = static_cast<float>((sx1 - fsx1) / cellWidth);
        }

        for (int sx = sx1; sx < sx2; sx++)
        {
            dtab[k] = dx;
            stab[k] = sx;
            vtab[k++] = static_cast<float>(1.0 / cellWidth);
        }

        if (fsx2 - sx2 > 1e-3)
        {
            dtab[k] = dx;
            stab[k] = sx2;
            vtab[k++] = static_cast<float>(std::min(std::min(fsx2 - sx2, 1.), cellWidth) / cellWidth);
        }
    }
    return k;
}

template<typename helper, int cn>
static inline int resizeAreaFast(int start, int end, const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step, int dst_width)
{
    using T = typename helper::ElemType;

    for (int i = start; i < end; i++)
    {
        int y_ofs0 = i * 2;
        int y_ofs1 = i * 2 + 1;

        int vl;
        switch (cn)
        {
        case 1:
            for (int j = 0; j < dst_width; j += vl)
            {
                vl = helper::setvl(dst_width - j);

                typename helper::VecType s0, s1;
                rvv<helper>::vlseg2e(reinterpret_cast<const T*>(src_data + y_ofs0 * src_step) + j * 2, vl, s0, s1);
                auto v0 = __riscv_vzext_vf2(s0, vl), v1 = __riscv_vzext_vf2(s1, vl);
                rvv<helper>::vlseg2e(reinterpret_cast<const T*>(src_data + y_ofs1 * src_step) + j * 2, vl, s0, s1);
                auto v2 = __riscv_vzext_vf2(s0, vl), v3 = __riscv_vzext_vf2(s1, vl);

                v0 = __riscv_vadd(__riscv_vadd(v0, v1, vl), __riscv_vadd(v2, v3, vl), vl);
                helper::vstore(reinterpret_cast<T*>(dst_data + i * dst_step) + j, __riscv_vnclipu(v0, 2, __RISCV_VXRM_RNU, vl), vl);
            }
            break;
        case 2:
            for (int j = 0; j < dst_width; j += vl)
            {
                vl = helper::setvl(dst_width - j);

                typename helper::VecType s0, s1;
                rvv<helper>::vlsseg2e(reinterpret_cast<const T*>(src_data + y_ofs0 * src_step) + j * 4, 4 * sizeof(T), vl, s0, s1);
                auto v00 = __riscv_vzext_vf2(s0, vl), v10 = __riscv_vzext_vf2(s1, vl);
                rvv<helper>::vlsseg2e(reinterpret_cast<const T*>(src_data + y_ofs0 * src_step) + j * 4 + 2, 4 * sizeof(T), vl, s0, s1);
                auto v01 = __riscv_vzext_vf2(s0, vl), v11 = __riscv_vzext_vf2(s1, vl);
                rvv<helper>::vlsseg2e(reinterpret_cast<const T*>(src_data + y_ofs1 * src_step) + j * 4, 4 * sizeof(T), vl, s0, s1);
                auto v02 = __riscv_vzext_vf2(s0, vl), v12 = __riscv_vzext_vf2(s1, vl);
                rvv<helper>::vlsseg2e(reinterpret_cast<const T*>(src_data + y_ofs1 * src_step) + j * 4 + 2, 4 * sizeof(T), vl, s0, s1);
                auto v03 = __riscv_vzext_vf2(s0, vl), v13 = __riscv_vzext_vf2(s1, vl);

                v00 = __riscv_vadd(__riscv_vadd(v00, v01, vl), __riscv_vadd(v02, v03, vl), vl);
                v10 = __riscv_vadd(__riscv_vadd(v10, v11, vl), __riscv_vadd(v12, v13, vl), vl);
                rvv<helper>::vsseg2e(reinterpret_cast<T*>(dst_data + i * dst_step) + j * 2, vl, __riscv_vnclipu(v00, 2, __RISCV_VXRM_RNU, vl), __riscv_vnclipu(v10, 2, __RISCV_VXRM_RNU, vl));
            }
            break;
        case 3:
            for (int j = 0; j < dst_width; j += vl)
            {
                vl = helper::setvl(dst_width - j);

                typename helper::VecType s0, s1, s2;
                rvv<helper>::vlsseg3e(reinterpret_cast<const T*>(src_data + y_ofs0 * src_step) + j * 6, 6 * sizeof(T), vl, s0, s1, s2);
                auto v00 = __riscv_vzext_vf2(s0, vl), v10 = __riscv_vzext_vf2(s1, vl), v20 = __riscv_vzext_vf2(s2, vl);
                rvv<helper>::vlsseg3e(reinterpret_cast<const T*>(src_data + y_ofs0 * src_step) + j * 6 + 3, 6 * sizeof(T), vl, s0, s1, s2);
                auto v01 = __riscv_vzext_vf2(s0, vl), v11 = __riscv_vzext_vf2(s1, vl), v21 = __riscv_vzext_vf2(s2, vl);
                rvv<helper>::vlsseg3e(reinterpret_cast<const T*>(src_data + y_ofs1 * src_step) + j * 6, 6 * sizeof(T), vl, s0, s1, s2);
                auto v02 = __riscv_vzext_vf2(s0, vl), v12 = __riscv_vzext_vf2(s1, vl), v22 = __riscv_vzext_vf2(s2, vl);
                rvv<helper>::vlsseg3e(reinterpret_cast<const T*>(src_data + y_ofs1 * src_step) + j * 6 + 3, 6 * sizeof(T), vl, s0, s1, s2);
                auto v03 = __riscv_vzext_vf2(s0, vl), v13 = __riscv_vzext_vf2(s1, vl), v23 = __riscv_vzext_vf2(s2, vl);

                v00 = __riscv_vadd(__riscv_vadd(v00, v01, vl), __riscv_vadd(v02, v03, vl), vl);
                v10 = __riscv_vadd(__riscv_vadd(v10, v11, vl), __riscv_vadd(v12, v13, vl), vl);
                v20 = __riscv_vadd(__riscv_vadd(v20, v21, vl), __riscv_vadd(v22, v23, vl), vl);
                rvv<helper>::vsseg3e(reinterpret_cast<T*>(dst_data + i * dst_step) + j * 3, vl, __riscv_vnclipu(v00, 2, __RISCV_VXRM_RNU, vl), __riscv_vnclipu(v10, 2, __RISCV_VXRM_RNU, vl), __riscv_vnclipu(v20, 2, __RISCV_VXRM_RNU, vl));
            }
            break;
        case 4:
            for (int j = 0; j < dst_width; j += vl)
            {
                vl = helper::setvl(dst_width - j);

                typename helper::VecType s0, s1, s2, s3;
                rvv<helper>::vlsseg4e(reinterpret_cast<const T*>(src_data + y_ofs0 * src_step) + j * 8, 8 * sizeof(T), vl, s0, s1, s2, s3);
                auto v00 = __riscv_vzext_vf2(s0, vl), v10 = __riscv_vzext_vf2(s1, vl), v20 = __riscv_vzext_vf2(s2, vl), v30 = __riscv_vzext_vf2(s3, vl);
                rvv<helper>::vlsseg4e(reinterpret_cast<const T*>(src_data + y_ofs0 * src_step) + j * 8 + 4, 8 * sizeof(T), vl, s0, s1, s2, s3);
                auto v01 = __riscv_vzext_vf2(s0, vl), v11 = __riscv_vzext_vf2(s1, vl), v21 = __riscv_vzext_vf2(s2, vl), v31 = __riscv_vzext_vf2(s3, vl);
                rvv<helper>::vlsseg4e(reinterpret_cast<const T*>(src_data + y_ofs1 * src_step) + j * 8, 8 * sizeof(T), vl, s0, s1, s2, s3);
                auto v02 = __riscv_vzext_vf2(s0, vl), v12 = __riscv_vzext_vf2(s1, vl), v22 = __riscv_vzext_vf2(s2, vl), v32 = __riscv_vzext_vf2(s3, vl);
                rvv<helper>::vlsseg4e(reinterpret_cast<const T*>(src_data + y_ofs1 * src_step) + j * 8 + 4, 8 * sizeof(T), vl, s0, s1, s2, s3);
                auto v03 = __riscv_vzext_vf2(s0, vl), v13 = __riscv_vzext_vf2(s1, vl), v23 = __riscv_vzext_vf2(s2, vl), v33 = __riscv_vzext_vf2(s3, vl);

                v00 = __riscv_vadd(__riscv_vadd(v00, v01, vl), __riscv_vadd(v02, v03, vl), vl);
                v10 = __riscv_vadd(__riscv_vadd(v10, v11, vl), __riscv_vadd(v12, v13, vl), vl);
                v20 = __riscv_vadd(__riscv_vadd(v20, v21, vl), __riscv_vadd(v22, v23, vl), vl);
                v30 = __riscv_vadd(__riscv_vadd(v30, v31, vl), __riscv_vadd(v32, v33, vl), vl);
                rvv<helper>::vsseg4e(reinterpret_cast<T*>(dst_data + i * dst_step) + j * 4, vl, __riscv_vnclipu(v00, 2, __RISCV_VXRM_RNU, vl), __riscv_vnclipu(v10, 2, __RISCV_VXRM_RNU, vl), __riscv_vnclipu(v20, 2, __RISCV_VXRM_RNU, vl), __riscv_vnclipu(v30, 2, __RISCV_VXRM_RNU, vl));
            }
            break;
        default:
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }
    }

    return CV_HAL_ERROR_OK;
}

template<int cn>
static inline int resizeArea(int start, int end, const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step, int dst_width, int xtab_size, int xtab_part, const ushort* x_stab, const ushort* x_dtab, const float* x_vtab, const ushort* y_stab, const ushort* y_dtab, const float* y_vtab, const ushort* tabofs)
{
    const int n = dst_width * cn;
    std::vector<float> _buf(n, 0), _sum(n, 0);
    float* buf = _buf.data(), *sum = _sum.data();

    start = tabofs[start];
    end = tabofs[end];
    int prev = y_dtab[start], vl;
    for (int i = start; i < end; i++)
    {
        memset(buf, 0, sizeof(float) * n);
        switch (cn)
        {
        case 1:
            for (int j = 0; j < xtab_part; j += vl)
            {
                vl = __riscv_vsetvl_e8m1(xtab_part - j);
                auto sxn = __riscv_vle16_v_u16m2(x_stab + j, vl);
                auto dxn = __riscv_vle16_v_u16m2(x_dtab + j, vl);
                auto vxn = __riscv_vle32_v_f32m4(x_vtab + j, vl);

                auto val = __riscv_vloxei16_v_f32m4(buf, dxn, vl);
                auto src = __riscv_vfcvt_f(__riscv_vzext_vf4(__riscv_vloxei16_v_u8m1(src_data + y_stab[i] * src_step, sxn, vl), vl), vl);
                __riscv_vsoxei16(buf, dxn, __riscv_vfmacc(val, src, vxn, vl), vl);
            }
            for (int j = xtab_part; j < xtab_size; j++)
            {
                buf[x_dtab[j]] += src_data[y_stab[i] * src_step + x_stab[j]] * x_vtab[j];
            }
            break;
        case 2:
            for (int j = 0; j < xtab_part; j += vl)
            {
                vl = __riscv_vsetvl_e8m1(xtab_part - j);
                auto sxn = __riscv_vle16_v_u16m2(x_stab + j, vl);
                auto dxn = __riscv_vle16_v_u16m2(x_dtab + j, vl);
                auto vxn = __riscv_vle32_v_f32m4(x_vtab + j, vl);

                auto val = __riscv_vloxseg2ei16_v_f32m4x2(buf, dxn, vl);
                auto src = __riscv_vloxseg2ei16_v_u8m1x2(src_data + y_stab[i] * src_step, sxn, vl);

                vfloat32m4x2_t dst{};
                dst = __riscv_vset_v_f32m4_f32m4x2(dst, 0, __riscv_vfmacc(__riscv_vget_v_f32m4x2_f32m4(val, 0), __riscv_vfcvt_f(__riscv_vzext_vf4(__riscv_vget_v_u8m1x2_u8m1(src, 0), vl), vl), vxn, vl));
                dst = __riscv_vset_v_f32m4_f32m4x2(dst, 1, __riscv_vfmacc(__riscv_vget_v_f32m4x2_f32m4(val, 1), __riscv_vfcvt_f(__riscv_vzext_vf4(__riscv_vget_v_u8m1x2_u8m1(src, 1), vl), vl), vxn, vl));
                __riscv_vsoxseg2ei16(buf, dxn, dst, vl);
            }
            for (int j = xtab_part; j < xtab_size; j++)
            {
                buf[x_dtab[j]    ] += src_data[y_stab[i] * src_step + x_stab[j]    ] * x_vtab[j];
                buf[x_dtab[j] + 1] += src_data[y_stab[i] * src_step + x_stab[j] + 1] * x_vtab[j];
            }
            break;
        case 3:
            for (int j = 0; j < xtab_part; j += vl)
            {
                vl = __riscv_vsetvl_e8mf2(xtab_part - j);
                auto sxn = __riscv_vle16_v_u16m1(x_stab + j, vl);
                auto dxn = __riscv_vle16_v_u16m1(x_dtab + j, vl);
                auto vxn = __riscv_vle32_v_f32m2(x_vtab + j, vl);

                auto val = __riscv_vloxseg3ei16_v_f32m2x3(buf, dxn, vl);
                auto src = __riscv_vloxseg3ei16_v_u8mf2x3(src_data + y_stab[i] * src_step, sxn, vl);

                vfloat32m2x3_t dst{};
                dst = __riscv_vset_v_f32m2_f32m2x3(dst, 0, __riscv_vfmacc(__riscv_vget_v_f32m2x3_f32m2(val, 0), __riscv_vfcvt_f(__riscv_vzext_vf4(__riscv_vget_v_u8mf2x3_u8mf2(src, 0), vl), vl), vxn, vl));
                dst = __riscv_vset_v_f32m2_f32m2x3(dst, 1, __riscv_vfmacc(__riscv_vget_v_f32m2x3_f32m2(val, 1), __riscv_vfcvt_f(__riscv_vzext_vf4(__riscv_vget_v_u8mf2x3_u8mf2(src, 1), vl), vl), vxn, vl));
                dst = __riscv_vset_v_f32m2_f32m2x3(dst, 2, __riscv_vfmacc(__riscv_vget_v_f32m2x3_f32m2(val, 2), __riscv_vfcvt_f(__riscv_vzext_vf4(__riscv_vget_v_u8mf2x3_u8mf2(src, 2), vl), vl), vxn, vl));
                __riscv_vsoxseg3ei16(buf, dxn, dst, vl);
            }
            for (int j = xtab_part; j < xtab_size; j++)
            {
                buf[x_dtab[j]    ] += src_data[y_stab[i] * src_step + x_stab[j]    ] * x_vtab[j];
                buf[x_dtab[j] + 1] += src_data[y_stab[i] * src_step + x_stab[j] + 1] * x_vtab[j];
                buf[x_dtab[j] + 2] += src_data[y_stab[i] * src_step + x_stab[j] + 2] * x_vtab[j];
            }
            break;
        case 4:
            for (int j = 0; j < xtab_part; j += vl)
            {
                vl = __riscv_vsetvl_e8mf2(xtab_part - j);
                auto sxn = __riscv_vle16_v_u16m1(x_stab + j, vl);
                auto dxn = __riscv_vle16_v_u16m1(x_dtab + j, vl);
                auto vxn = __riscv_vle32_v_f32m2(x_vtab + j, vl);

                auto val = __riscv_vloxseg4ei16_v_f32m2x4(buf, dxn, vl);
                auto src = __riscv_vloxseg4ei16_v_u8mf2x4(src_data + y_stab[i] * src_step, sxn, vl);

                vfloat32m2x4_t dst{};
                dst = __riscv_vset_v_f32m2_f32m2x4(dst, 0, __riscv_vfmacc(__riscv_vget_v_f32m2x4_f32m2(val, 0), __riscv_vfcvt_f(__riscv_vzext_vf4(__riscv_vget_v_u8mf2x4_u8mf2(src, 0), vl), vl), vxn, vl));
                dst = __riscv_vset_v_f32m2_f32m2x4(dst, 1, __riscv_vfmacc(__riscv_vget_v_f32m2x4_f32m2(val, 1), __riscv_vfcvt_f(__riscv_vzext_vf4(__riscv_vget_v_u8mf2x4_u8mf2(src, 1), vl), vl), vxn, vl));
                dst = __riscv_vset_v_f32m2_f32m2x4(dst, 2, __riscv_vfmacc(__riscv_vget_v_f32m2x4_f32m2(val, 2), __riscv_vfcvt_f(__riscv_vzext_vf4(__riscv_vget_v_u8mf2x4_u8mf2(src, 2), vl), vl), vxn, vl));
                dst = __riscv_vset_v_f32m2_f32m2x4(dst, 3, __riscv_vfmacc(__riscv_vget_v_f32m2x4_f32m2(val, 3), __riscv_vfcvt_f(__riscv_vzext_vf4(__riscv_vget_v_u8mf2x4_u8mf2(src, 3), vl), vl), vxn, vl));
                __riscv_vsoxseg4ei16(buf, dxn, dst, vl);
            }
            for (int j = xtab_part; j < xtab_size; j++)
            {
                buf[x_dtab[j]    ] += src_data[y_stab[i] * src_step + x_stab[j]    ] * x_vtab[j];
                buf[x_dtab[j] + 1] += src_data[y_stab[i] * src_step + x_stab[j] + 1] * x_vtab[j];
                buf[x_dtab[j] + 2] += src_data[y_stab[i] * src_step + x_stab[j] + 2] * x_vtab[j];
                buf[x_dtab[j] + 3] += src_data[y_stab[i] * src_step + x_stab[j] + 3] * x_vtab[j];
            }
            break;
        default:
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        if (y_dtab[i] != prev)
        {
            for (int j = 0; j < n; j += vl)
            {
                vl = __riscv_vsetvl_e32m4(n - j);
                auto val = __riscv_vle32_v_f32m4(sum + j, vl);
                __riscv_vse8(dst_data + prev * dst_step + j, __riscv_vnclipu(__riscv_vfncvt_xu(val, vl), 0, __RISCV_VXRM_RNU, vl), vl);
                __riscv_vse32(sum + j, __riscv_vfmul(__riscv_vle32_v_f32m4(buf + j, vl), y_vtab[i], vl), vl);
            }
            prev = y_dtab[i];
        }
        else
        {
            for (int j = 0; j < n; j += vl)
            {
                vl = __riscv_vsetvl_e32m4(n - j);
                __riscv_vse32(sum + j, __riscv_vfmacc(__riscv_vle32_v_f32m4(sum + j, vl), y_vtab[i], __riscv_vle32_v_f32m4(buf + j, vl), vl), vl);
            }
        }
    }
    for (int j = 0; j < n; j += vl)
    {
        vl = __riscv_vsetvl_e32m4(n - j);
        auto val = __riscv_vle32_v_f32m4(sum + j, vl);
        __riscv_vse8(dst_data + prev * dst_step + j, __riscv_vnclipu(__riscv_vfncvt_xu(val, vl), 0, __RISCV_VXRM_RNU, vl), vl);
    }

    return CV_HAL_ERROR_OK;
}

// the algorithm is copied from imgproc/src/resize.cpp,
// in the function static void resizeNN and static void resizeNN_bitexact
static inline int resizeNN(int src_type, const uchar *src_data, size_t src_step, int src_width, int src_height, uchar *dst_data, size_t dst_step, int dst_width, int dst_height, double scale_x, double scale_y, int interpolation)
{
    const int cn = CV_ELEM_SIZE(src_type);
    if (cn * src_width > std::numeric_limits<ushort>::max())
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    std::vector<ushort> x_ofs(dst_width);
    const int ifx = ((src_width << 16) + dst_width / 2) / dst_width;
    const int ifx0 = ifx / 2 - src_width % 2;
    for (int i = 0; i < dst_width; i++)
    {
        x_ofs[i] = interpolation == CV_HAL_INTER_NEAREST ? static_cast<ushort>(std::floor(i * scale_x)) : (ifx * i + ifx0) >> 16;
        x_ofs[i] = std::min(x_ofs[i], static_cast<ushort>(src_width - 1)) * cn;
    }

    switch (src_type)
    {
    case CV_8UC1:
        return invoke(dst_height, {resizeNN<1>}, src_data, src_step, src_height, dst_data, dst_step, dst_width, dst_height, scale_y, interpolation, x_ofs.data());
    case CV_8UC2:
        return invoke(dst_height, {resizeNN<2>}, src_data, src_step, src_height, dst_data, dst_step, dst_width, dst_height, scale_y, interpolation, x_ofs.data());
    case CV_8UC3:
        return invoke(dst_height, {resizeNN<3>}, src_data, src_step, src_height, dst_data, dst_step, dst_width, dst_height, scale_y, interpolation, x_ofs.data());
    case CV_8UC4:
        return invoke(dst_height, {resizeNN<4>}, src_data, src_step, src_height, dst_data, dst_step, dst_width, dst_height, scale_y, interpolation, x_ofs.data());
    }
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

// the algorithm is copied from imgproc/src/resize.cpp,
// in the functor HResizeLinear, VResizeLinear and resize_bitExact
static inline int resizeLinear(int src_type, const uchar *src_data, size_t src_step, int src_width, int src_height, uchar *dst_data, size_t dst_step, int dst_width, int dst_height, double scale_x, double scale_y, int interpolation)
{
    const int cn = CV_ELEM_SIZE(src_type);
    if (cn * src_width > std::numeric_limits<ushort>::max())
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    std::vector<ushort> x_ofs0(dst_width), x_ofs1(dst_width);
    if (interpolation == CV_HAL_INTER_LINEAR_EXACT)
    {
        std::vector<ushort> x_val(dst_width);
        for (int i = 0; i < dst_width; i++)
        {
            double val = (i + 0.5) * scale_x - 0.5;
            int x_ofs = static_cast<int>(std::floor(val));
            val -= x_ofs;

            x_val[i] = static_cast<ushort>(val * 256 - std::remainder(val * 256, 1));
            x_ofs0[i] = static_cast<ushort>(std::min(std::max(x_ofs    , 0), src_width - 1)) * cn;
            x_ofs1[i] = static_cast<ushort>(std::min(std::max(x_ofs + 1, 0), src_width - 1)) * cn;
        }

        switch (src_type)
        {
        case CV_8UC1:
            return invoke(dst_height, {resizeLinearExact<1>}, src_data, src_step, src_height, dst_data, dst_step, dst_width, scale_y, x_ofs0.data(), x_ofs1.data(), x_val.data());
        case CV_8UC2:
            return invoke(dst_height, {resizeLinearExact<2>}, src_data, src_step, src_height, dst_data, dst_step, dst_width, scale_y, x_ofs0.data(), x_ofs1.data(), x_val.data());
        case CV_8UC3:
            return invoke(dst_height, {resizeLinearExact<3>}, src_data, src_step, src_height, dst_data, dst_step, dst_width, scale_y, x_ofs0.data(), x_ofs1.data(), x_val.data());
        case CV_8UC4:
            return invoke(dst_height, {resizeLinearExact<4>}, src_data, src_step, src_height, dst_data, dst_step, dst_width, scale_y, x_ofs0.data(), x_ofs1.data(), x_val.data());
        }
    }
    else
    {
        std::vector<float> x_val(dst_width);
        for (int i = 0; i < dst_width; i++)
        {
            x_val[i] = (i + 0.5) * scale_x - 0.5;
            int x_ofs = static_cast<int>(std::floor(x_val[i]));
            x_val[i] -= x_ofs;

            x_ofs0[i] = static_cast<ushort>(std::min(std::max(x_ofs    , 0), src_width - 1)) * cn;
            x_ofs1[i] = static_cast<ushort>(std::min(std::max(x_ofs + 1, 0), src_width - 1)) * cn;
        }

        switch (src_type)
        {
        case CV_8UC1:
            return invoke(dst_height, {resizeLinear<RVV_U8M1, 1>}, src_data, src_step, src_height, dst_data, dst_step, dst_width, scale_y, x_ofs0.data(), x_ofs1.data(), x_val.data());
        case CV_16UC1:
            return invoke(dst_height, {resizeLinear<RVV_U16M2, 1>}, src_data, src_step, src_height, dst_data, dst_step, dst_width, scale_y, x_ofs0.data(), x_ofs1.data(), x_val.data());
        case CV_32FC1:
            return invoke(dst_height, {resizeLinear<RVV_F32M4, 1>}, src_data, src_step, src_height, dst_data, dst_step, dst_width, scale_y, x_ofs0.data(), x_ofs1.data(), x_val.data());
        case CV_8UC2:
            return invoke(dst_height, {resizeLinear<RVV_U8M1, 2>}, src_data, src_step, src_height, dst_data, dst_step, dst_width, scale_y, x_ofs0.data(), x_ofs1.data(), x_val.data());
        case CV_16UC2:
            return invoke(dst_height, {resizeLinear<RVV_U16M2, 2>}, src_data, src_step, src_height, dst_data, dst_step, dst_width, scale_y, x_ofs0.data(), x_ofs1.data(), x_val.data());
        case CV_32FC2:
            return invoke(dst_height, {resizeLinear<RVV_F32M4, 2>}, src_data, src_step, src_height, dst_data, dst_step, dst_width, scale_y, x_ofs0.data(), x_ofs1.data(), x_val.data());

        case CV_8UC3:
            return invoke(dst_height, {resizeLinear<RVV_U8MF2, 3>}, src_data, src_step, src_height, dst_data, dst_step, dst_width, scale_y, x_ofs0.data(), x_ofs1.data(), x_val.data());
        case CV_16UC3:
            return invoke(dst_height, {resizeLinear<RVV_U16M1, 3>}, src_data, src_step, src_height, dst_data, dst_step, dst_width, scale_y, x_ofs0.data(), x_ofs1.data(), x_val.data());
        case CV_32FC3:
            return invoke(dst_height, {resizeLinear<RVV_F32M2, 3>}, src_data, src_step, src_height, dst_data, dst_step, dst_width, scale_y, x_ofs0.data(), x_ofs1.data(), x_val.data());
        case CV_8UC4:
            return invoke(dst_height, {resizeLinear<RVV_U8MF2, 4>}, src_data, src_step, src_height, dst_data, dst_step, dst_width, scale_y, x_ofs0.data(), x_ofs1.data(), x_val.data());
        case CV_16UC4:
            return invoke(dst_height, {resizeLinear<RVV_U16M1, 4>}, src_data, src_step, src_height, dst_data, dst_step, dst_width, scale_y, x_ofs0.data(), x_ofs1.data(), x_val.data());
        case CV_32FC4:
            return invoke(dst_height, {resizeLinear<RVV_F32M2, 4>}, src_data, src_step, src_height, dst_data, dst_step, dst_width, scale_y, x_ofs0.data(), x_ofs1.data(), x_val.data());
        }
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

// the algorithm is copied from imgproc/src/resize.cpp,
// in the function template static void resizeArea_ and template static void resizeAreaFast_
static inline int resizeArea(int src_type, const uchar *src_data, size_t src_step, int src_width, int src_height, uchar *dst_data, size_t dst_step, int dst_width, int dst_height, double scale_x, double scale_y)
{
    if (scale_x < 1 || scale_y < 1)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if (dst_width * 2 == src_width && dst_height * 2 == src_height)
    {
        switch (src_type)
        {
        case CV_8UC1:
            return invoke(dst_height, {resizeAreaFast<RVV_U8M1, 1>}, src_data, src_step, dst_data, dst_step, dst_width);
        case CV_16UC1:
            return invoke(dst_height, {resizeAreaFast<RVV_U16M2, 1>}, src_data, src_step, dst_data, dst_step, dst_width);
        case CV_8UC2:
            return invoke(dst_height, {resizeAreaFast<RVV_U8M1, 2>}, src_data, src_step, dst_data, dst_step, dst_width);
        case CV_16UC2:
            return invoke(dst_height, {resizeAreaFast<RVV_U16M2, 2>}, src_data, src_step, dst_data, dst_step, dst_width);
        case CV_8UC3:
            return invoke(dst_height, {resizeAreaFast<RVV_U8M1, 3>}, src_data, src_step, dst_data, dst_step, dst_width);
        case CV_16UC3:
            return invoke(dst_height, {resizeAreaFast<RVV_U16M2, 3>}, src_data, src_step, dst_data, dst_step, dst_width);
        case CV_8UC4:
            return invoke(dst_height, {resizeAreaFast<RVV_U8M1, 4>}, src_data, src_step, dst_data, dst_step, dst_width);
        case CV_16UC4:
            return invoke(dst_height, {resizeAreaFast<RVV_U16M2, 4>}, src_data, src_step, dst_data, dst_step, dst_width);
        }
    }
    else
    {
        const int cn = CV_ELEM_SIZE(src_type);
        if (cn * sizeof(float) * src_width > std::numeric_limits<ushort>::max())
            return CV_HAL_ERROR_NOT_IMPLEMENTED;

        std::vector<ushort> x_stab(src_width * 2), x_dtab(src_width * 2), y_stab(src_height * 2), y_dtab(src_height * 2);
        std::vector<float> x_vtab(src_width * 2), y_vtab(src_height * 2);
        int xtab_size = computeResizeAreaTab(src_width , dst_width , scale_x, x_stab.data(), x_dtab.data(), x_vtab.data());
        int ytab_size = computeResizeAreaTab(src_height, dst_height, scale_y, y_stab.data(), y_dtab.data(), y_vtab.data());

        // reorder xtab to avoid data dependency between __riscv_vloxei and __riscv_vsoxei
        int range = __riscv_vlenb() * 4 / sizeof(float);
        std::vector<std::vector<ushort>> idx(dst_width);
        for (int i = 0; i < xtab_size; i++)
        {
            idx[x_dtab[i]].push_back(i);
        }
        std::list<ushort> remain;
        for (int i = 0; i < dst_width; i++)
        {
            remain.push_back(i);
        }

        std::vector<ushort> list;
        for (int i = 0; i < xtab_size / range; i++)
        {
            auto it = remain.begin();
            int j;
            for (j = 0; j < range; j++)
            {
                if (it == remain.end())
                    break;
                ushort val = *it;

                list.push_back(idx[val].back());
                idx[val].pop_back();
                it = idx[val].empty() ? remain.erase(it) : ++it;
            }
            if (j < range)
                break;
        }

        int xtab_part = list.size();
        for (auto val : remain)
        {
            for (ushort id : idx[val])
            {
                list.push_back(id);
            }
        }
        for (int i = 0; i < xtab_size; i++)
        {
            ushort stab = x_stab[i], dtab = x_dtab[i];
            float vtab = x_vtab[i];
            int j = i;
            while (true)
            {
                int k = list[j];
                list[j] = j;
                if (k == i)
                    break;
                x_stab[j] = x_stab[k];
                x_dtab[j] = x_dtab[k];
                x_vtab[j] = x_vtab[k];
                j = k;
            }
            x_stab[j] = stab;
            x_dtab[j] = dtab;
            x_vtab[j] = vtab;
        }
        for (int i = 0; i < xtab_size; i++)
        {
            x_stab[i] *= cn;
            x_dtab[i] *= cn;
            if (i < xtab_part)
            {
                x_dtab[i] *= sizeof(float);
            }
        }
        // reorder done

        std::vector<ushort> tabofs;
        for (int i = 0; i < ytab_size; i++)
        {
            if (i == 0 || y_dtab[i] != y_dtab[i - 1])
            {
                tabofs.push_back(i);
            }
        }
        tabofs.push_back(ytab_size);

        switch (src_type)
        {
        case CV_8UC1:
            return invoke(dst_height, {resizeArea<1>}, src_data, src_step, dst_data, dst_step, dst_width, xtab_size, xtab_part, x_stab.data(), x_dtab.data(), x_vtab.data(), y_stab.data(), y_dtab.data(), y_vtab.data(), tabofs.data());
        case CV_8UC2:
            return invoke(dst_height, {resizeArea<2>}, src_data, src_step, dst_data, dst_step, dst_width, xtab_size, xtab_part, x_stab.data(), x_dtab.data(), x_vtab.data(), y_stab.data(), y_dtab.data(), y_vtab.data(), tabofs.data());
        case CV_8UC3:
            return invoke(dst_height, {resizeArea<3>}, src_data, src_step, dst_data, dst_step, dst_width, xtab_size, xtab_part, x_stab.data(), x_dtab.data(), x_vtab.data(), y_stab.data(), y_dtab.data(), y_vtab.data(), tabofs.data());
        case CV_8UC4:
            return invoke(dst_height, {resizeArea<4>}, src_data, src_step, dst_data, dst_step, dst_width, xtab_size, xtab_part, x_stab.data(), x_dtab.data(), x_vtab.data(), y_stab.data(), y_dtab.data(), y_vtab.data(), tabofs.data());
        }
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

inline int resize(int src_type, const uchar *src_data, size_t src_step, int src_width, int src_height, uchar *dst_data, size_t dst_step, int dst_width, int dst_height, double inv_scale_x, double inv_scale_y, int interpolation)
{
    inv_scale_x = 1 / inv_scale_x;
    inv_scale_y = 1 / inv_scale_y;
    if (interpolation == CV_HAL_INTER_NEAREST || interpolation == CV_HAL_INTER_NEAREST_EXACT)
        return resizeNN(src_type, src_data, src_step, src_width, src_height, dst_data, dst_step, dst_width, dst_height, inv_scale_x, inv_scale_y, interpolation);
    if (interpolation == CV_HAL_INTER_LINEAR || interpolation == CV_HAL_INTER_LINEAR_EXACT)
        return resizeLinear(src_type, src_data, src_step, src_width, src_height, dst_data, dst_step, dst_width, dst_height, inv_scale_x, inv_scale_y, interpolation);
    if (interpolation == CV_HAL_INTER_AREA)
        return resizeArea(src_type, src_data, src_step, src_width, src_height, dst_data, dst_step, dst_width, dst_height, inv_scale_x, inv_scale_y);

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}
} // cv::cv_hal_rvv::resize

}}

#endif
