// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copytright (C) 2025, SpaceMIT Inc., all rights reserved.

#include "opencv2/core/hal/intrin.hpp"

namespace cv {

namespace {

// [TODO] Drop this until rvv has dedicated intrinsics for abs on integers.
template<typename T, typename ST> inline ST __riscv_vabs(const T&);

template<> inline
vuint8m1_t __riscv_vabs(const vint8m1_t& v) {
    const int vle8m1 = __riscv_vsetvlmax_e8m1();
    vint8m1_t mask = __riscv_vsra_vx_i8m1(v, 7, vle8m1);
    vint8m1_t v_xor = __riscv_vxor_vv_i8m1(v, mask, vle8m1);
    return __riscv_vreinterpret_v_i8m1_u8m1(
        __riscv_vsub_vv_i8m1(v_xor, mask, vle8m1)
    );
}

template<> inline
vuint16m1_t __riscv_vabs(const vint16m1_t& v) {
    const int vle16m1 = __riscv_vsetvlmax_e16m1();
    vint16m1_t mask = __riscv_vsra_vx_i16m1(v, 15, vle16m1);
    vint16m1_t v_xor = __riscv_vxor_vv_i16m1(v, mask, vle16m1);
    return __riscv_vreinterpret_v_i16m1_u16m1(
        __riscv_vsub_vv_i16m1(v_xor, mask, vle16m1)
    );
}

template<> inline
vuint32m1_t __riscv_vabs(const vint32m1_t& v) {
    const int vle32m1 = __riscv_vsetvlmax_e32m1();
    vint32m1_t mask = __riscv_vsra_vx_i32m1(v, 31, vle32m1);
    vint32m1_t v_xor = __riscv_vxor_vv_i32m1(v, mask, vle32m1);
    return __riscv_vreinterpret_v_i32m1_u32m1(
        __riscv_vsub_vv_i32m1(v_xor, mask, vle32m1)
    );
}

// [TODO] Drop this until rvv has dedicated intrinsics for absdiff on integers.
template<typename T, typename ST> inline ST __riscv_vabsdiff(const T&, const T&);

template<> inline
vuint8m1_t __riscv_vabsdiff(const vint8m1_t& src1, const vint8m1_t& src2) {
    const int vle8m1 = __riscv_vsetvlmax_e8m1();
    vint8m1_t ab_max = __riscv_vmax(src1, src2, vle8m1);
    vint8m1_t ab_min = __riscv_vmin(src1, src2, vle8m1);
    return __riscv_vreinterpret_v_i8m1_u8m1(
        __riscv_vsub(ab_max, ab_min, vle8m1)
    );
}

template<> inline
vuint16m1_t __riscv_vabsdiff(const vuint16m1_t& src1, const vuint16m1_t& src2) {
    const int vle16m1 = __riscv_vsetvlmax_e16m1();
    vuint16m1_t ab_max = __riscv_vmaxu(src1, src2, vle16m1);
    vuint16m1_t ab_min = __riscv_vminu(src1, src2, vle16m1);
    return __riscv_vsub(ab_max, ab_min, vle16m1);
}

template<> inline
vuint16m1_t __riscv_vabsdiff(const vint16m1_t& src1, const vint16m1_t& src2) {
    const int vle16m1 = __riscv_vsetvlmax_e16m1();
    vint16m1_t ab_max = __riscv_vmax(src1, src2, vle16m1);
    vint16m1_t ab_min = __riscv_vmin(src1, src2, vle16m1);
    return __riscv_vreinterpret_v_i16m1_u16m1(__riscv_vsub(ab_max, ab_min, vle16m1));
}

template<> inline
vuint32m1_t __riscv_vabsdiff(const vint32m1_t& src1, const vint32m1_t& src2) {
    const int vle32m1 = __riscv_vsetvlmax_e32m1();
    vint32m1_t ab_max = __riscv_vmax(src1, src2, vle32m1);
    vint32m1_t ab_min = __riscv_vmin(src1, src2, vle32m1);
    return __riscv_vreinterpret_v_i32m1_u32m1(__riscv_vsub(ab_max, ab_min, vle32m1));
}

}

CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

template <typename T, typename ST> inline
ST normInf_rvv(const T* src, int n, int& j);

template<> inline
int normInf_rvv(const int* src, int n, int& j) {
    const int vle32m1 = __riscv_vsetvlmax_e32m1();
    vuint32m1_t r0 = __riscv_vmv_v_x_u32m1(0, vle32m1);
    vuint32m1_t r1 = __riscv_vmv_v_x_u32m1(0, vle32m1);
    for (; j <= n - 2 * vle32m1; j += 2 * vle32m1) {
        vuint32m1_t v0 = __riscv_vabs<vint32m1_t, vuint32m1_t>(__riscv_vle32_v_i32m1(src + j, vle32m1));
        r0 = __riscv_vmaxu(r0, v0, vle32m1);

        vuint32m1_t v1 = __riscv_vabs<vint32m1_t, vuint32m1_t>(__riscv_vle32_v_i32m1(src + j + vle32m1, vle32m1));
        r1 = __riscv_vmaxu(r1, v1, vle32m1);
    }
    r0 = __riscv_vmaxu(r0, r1, vle32m1);
    return (int)__riscv_vmv_x(__riscv_vredmaxu(r0, __riscv_vmv_v_x_u32m1(0, vle32m1), vle32m1));
}

template <typename T, typename ST> inline
ST normL1_rvv(const T* src, int n, int& j);

template<> inline
int normL1_rvv(const schar* src, int n, int& j) {
    const int vle8m1 = __riscv_vsetvlmax_e8m1();
    const int vle16m1 = __riscv_vsetvlmax_e16m1();
    const int vle32m1 = __riscv_vsetvlmax_e32m1();
    vuint32m1_t r0 = __riscv_vmv_v_x_u32m1(0, vle32m1);
    vuint32m1_t r1 = __riscv_vmv_v_x_u32m1(0, vle32m1);
    vuint16m1_t zero = __riscv_vmv_v_x_u16m1(0, vle16m1);
    for (; j <= n - 2 * vle8m1; j += 2 * vle8m1) {
        vuint8m1_t v0 = __riscv_vabs<vint8m1_t, vuint8m1_t>(__riscv_vle8_v_i8m1(src + j, vle8m1));
        vuint16m1_t u0 = __riscv_vwredsumu_tu(zero, v0, zero, vle8m1);
        r0 = __riscv_vwredsumu(u0, r0, vle16m1);

        vuint8m1_t v1 = __riscv_vabs<vint8m1_t, vuint8m1_t>(__riscv_vle8_v_i8m1(src + j + vle8m1, vle8m1));
        vuint16m1_t u1 = __riscv_vwredsumu_tu(zero, v1, zero, vle8m1);
        r1 = __riscv_vwredsumu(u1, r1, vle16m1);
    }
    return (int)__riscv_vmv_x(__riscv_vadd(r0, r1, vle32m1));
}

template<> inline
int normL1_rvv(const ushort* src, int n, int& j) {
    const int vle16m1 = __riscv_vsetvlmax_e16m1();
    const int vle32m1 = __riscv_vsetvlmax_e32m1();
    vuint32m1_t r0 = __riscv_vmv_v_x_u32m1(0, vle32m1);
    vuint32m1_t r1 = __riscv_vmv_v_x_u32m1(0, vle32m1);
    for (; j <= n - 2 * vle16m1; j += 2 * vle16m1) {
        vuint16m1_t v0 = __riscv_vle16_v_u16m1(src + j, vle16m1);
        r0 = __riscv_vwredsumu(v0, r0, vle16m1);

        vuint16m1_t v1 = __riscv_vle16_v_u16m1(src + j + vle16m1, vle16m1);
        r1 = __riscv_vwredsumu(v1, r1, vle16m1);
    }
    return (int)__riscv_vmv_x(__riscv_vadd(r0, r1, vle32m1));
}

template<> inline
int normL1_rvv(const short* src, int n, int& j) {
    const int vle16m1 = __riscv_vsetvlmax_e16m1();
    const int vle32m1 = __riscv_vsetvlmax_e32m1();
    vuint32m1_t r0 = __riscv_vmv_v_x_u32m1(0, vle32m1);
    vuint32m1_t r1 = __riscv_vmv_v_x_u32m1(0, vle32m1);
    for (; j<= n - 2 * vle16m1; j += 2 * vle16m1) {
        vuint16m1_t v0 = __riscv_vabs<vint16m1_t, vuint16m1_t>(__riscv_vle16_v_i16m1(src + j, vle16m1));
        r0 = __riscv_vwredsumu(v0, r0, vle16m1);

        vuint16m1_t v1 = __riscv_vabs<vint16m1_t, vuint16m1_t>(__riscv_vle16_v_i16m1(src + j + vle16m1, vle16m1));
        r1 = __riscv_vwredsumu(v1, r1, vle16m1);
    }
    return (int)__riscv_vmv_x(__riscv_vadd(r0, r1, vle32m1));
}

template<> inline
double normL1_rvv(const double* src, int n, int& j) {
    const int vle64m1 = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t r0 = __riscv_vfmv_v_f_f64m1(0.f, vle64m1);
    vfloat64m1_t r1 = __riscv_vfmv_v_f_f64m1(0.f, vle64m1);
    for (; j <= n - 2 * vle64m1; j += 2 * vle64m1) {
        vfloat64m1_t v0 = __riscv_vle64_v_f64m1(src + j, vle64m1);
        v0 = __riscv_vfabs(v0, vle64m1);
        r0 = __riscv_vfadd(r0, v0, vle64m1);

        vfloat64m1_t v1 = __riscv_vle64_v_f64m1(src + j + vle64m1, vle64m1);
        v1 = __riscv_vfabs(v1, vle64m1);
        r1 = __riscv_vfadd(r1, v1, vle64m1);
    }
    r0 = __riscv_vfadd(r0, r1, vle64m1);
    return __riscv_vfmv_f(__riscv_vfredusum(r0, __riscv_vfmv_v_f_f64m1(0.f, vle64m1), vle64m1));
}

template <typename T, typename ST> inline
ST normL2_rvv(const T* src, int n, int& j);

template<> inline
int normL2_rvv(const uchar* src, int n, int& j) {
    const int vle8m1 = __riscv_vsetvlmax_e8m1();
    const int vle16m1 = __riscv_vsetvlmax_e16m1();
    const int vle32m1 = __riscv_vsetvlmax_e32m1();
    vuint32m1_t r0 = __riscv_vmv_v_x_u32m1(0, vle32m1);
    vuint32m1_t r1 = __riscv_vmv_v_x_u32m1(0, vle32m1);
    for (; j <= n - 2 * vle8m1; j += 2 * vle8m1) {
        vuint8m1_t v0 = __riscv_vle8_v_u8m1(src + j, vle8m1);
        vuint16m2_t u0 = __riscv_vwmulu(v0, v0, vle8m1);
        r0 = __riscv_vwredsumu(u0, r0, vle16m1 * 2);

        vuint8m1_t v1 = __riscv_vle8_v_u8m1(src + j + vle8m1, vle8m1);
        vuint16m2_t u1 = __riscv_vwmulu(v1, v1, vle8m1);
        r1 = __riscv_vwredsumu(u1, r1, vle16m1 * 2);
    }
    return (int)__riscv_vmv_x(__riscv_vadd(r0, r1, vle32m1));
}

template<> inline
int normL2_rvv(const schar* src, int n, int& j) {
    const int vle8m1 = __riscv_vsetvlmax_e8m1();
    const int vle16m1 = __riscv_vsetvlmax_e16m1();
    const int vle32m1 = __riscv_vsetvlmax_e32m1();
    vint32m1_t r0 = __riscv_vmv_v_x_i32m1(0, vle32m1);
    vint32m1_t r1 = __riscv_vmv_v_x_i32m1(0, vle32m1);
    for (; j <= n - 2 * vle8m1; j += 2 * vle8m1) {
        vint8m1_t v0 = __riscv_vle8_v_i8m1(src + j, vle8m1);
        vint16m2_t u0 = __riscv_vwmul(v0, v0, vle8m1);
        r0 = __riscv_vwredsum(u0, r0, vle16m1 * 2);

        vint8m1_t v1 = __riscv_vle8_v_i8m1(src + j + vle8m1, vle8m1);
        vint16m2_t u1 = __riscv_vwmul(v1, v1, vle8m1);
        r1 = __riscv_vwredsum(u1, r1, vle16m1 * 2);
    }
    return __riscv_vmv_x(__riscv_vadd(r0, r1, vle32m1));
}

template<> inline
double normL2_rvv(const double* src, int n, int& j) {
    const int vle64m1 = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t r0 = __riscv_vfmv_v_f_f64m1(0.f, vle64m1);
    vfloat64m1_t r1 = __riscv_vfmv_v_f_f64m1(0.f, vle64m1);
    for (; j <= n - 2 * vle64m1; j += 2 * vle64m1) {
        vfloat64m1_t v0 = __riscv_vle64_v_f64m1(src + j, vle64m1);
        r0 = __riscv_vfmacc(r0, v0, v0, vle64m1);

        vfloat64m1_t v1 = __riscv_vle64_v_f64m1(src + j + vle64m1, vle64m1);
        r1 = __riscv_vfmacc(r1, v1, v1, vle64m1);
    }
    r0 = __riscv_vfadd(r0, r1, vle64m1);
    return __riscv_vfmv_f(__riscv_vfredusum(r0, __riscv_vfmv_v_f_f64m1(0.f, vle64m1), vle64m1));
}

template <typename T, typename ST> inline
ST normDiffInf_rvv(const T* src1, const T* src2, int n, int& j);

template<> inline
int normDiffInf_rvv(const schar* src1, const schar* src2, int n, int& j) {
    const int vle8m1 = __riscv_vsetvlmax_e8m1();
    vuint8m1_t r0 = __riscv_vmv_v_x_u8m1(0, vle8m1);
    vuint8m1_t r1 = __riscv_vmv_v_x_u8m1(0, vle8m1);
    for (; j <= n - 2 * vle8m1; j += 2 * vle8m1) {
        vint8m1_t v01 = __riscv_vle8_v_i8m1(src1 + j, vle8m1),
                  v02 = __riscv_vle8_v_i8m1(src2 + j, vle8m1);
        vuint8m1_t v0 = __riscv_vabsdiff<vint8m1_t, vuint8m1_t>(v01, v02);
        r0 = __riscv_vmaxu(r0, v0, vle8m1);

        vint8m1_t v11 = __riscv_vle8_v_i8m1(src1 + j + vle8m1, vle8m1),
                  v12 = __riscv_vle8_v_i8m1(src2 + j + vle8m1, vle8m1);
        vuint8m1_t v1 = __riscv_vabsdiff<vint8m1_t, vuint8m1_t>(v11, v12);
        r1 = __riscv_vmaxu(r1, v1, vle8m1);
    }
    return (int)__riscv_vmv_x(__riscv_vredmaxu(__riscv_vmaxu(r0, r1, vle8m1), r0, vle8m1));
}

template<> inline
int normDiffInf_rvv(const short* src1, const short* src2, int n, int& j) {
    const int vle16m1 = __riscv_vsetvlmax_e16m1();
    vuint16m1_t r0 = __riscv_vmv_v_x_u16m1(0, vle16m1);
    vuint16m1_t r1 = __riscv_vmv_v_x_u16m1(0, vle16m1);
    for (; j <= n - 2 * vle16m1; j += 2 * vle16m1) {
        vint16m1_t v01 = __riscv_vle16_v_i16m1(src1 + j, vle16m1),
                   v02 = __riscv_vle16_v_i16m1(src2 + j, vle16m1);
        vuint16m1_t v0 = __riscv_vabsdiff<vint16m1_t, vuint16m1_t>(v01, v02);
        r0 = __riscv_vmaxu(r0, v0, vle16m1);

        vint16m1_t v11 = __riscv_vle16_v_i16m1(src1 + j + vle16m1, vle16m1),
                   v12 = __riscv_vle16_v_i16m1(src2 + j + vle16m1, vle16m1);
        vuint16m1_t v1 = __riscv_vabsdiff<vint16m1_t, vuint16m1_t>(v11, v12);
        r1 = __riscv_vmaxu(r1, v1, vle16m1);
    }
    return (int)__riscv_vmv_x(__riscv_vredmaxu(__riscv_vmaxu(r0, r1, vle16m1), r0, vle16m1));
}

template<> inline
int normDiffInf_rvv(const int* src1, const int* src2, int n, int& j) {
    const int vle32m1 = __riscv_vsetvlmax_e32m1();
    vuint32m1_t r0 = __riscv_vmv_v_x_u32m1(0, vle32m1);
    vuint32m1_t r1 = __riscv_vmv_v_x_u32m1(0, vle32m1);
    for (; j <= n - 2 * vle32m1; j += 2 * vle32m1) {
        vint32m1_t v01 = __riscv_vle32_v_i32m1(src1 + j, vle32m1),
                   v02 = __riscv_vle32_v_i32m1(src2 + j, vle32m1);
        vuint32m1_t v0 = __riscv_vabsdiff<vint32m1_t, vuint32m1_t>(v01, v02);
        r0 = __riscv_vmaxu(r0, v0, vle32m1);

        vint32m1_t v11 = __riscv_vle32_v_i32m1(src1 + j + vle32m1, vle32m1),
                   v12 = __riscv_vle32_v_i32m1(src2 + j + vle32m1, vle32m1);
        vuint32m1_t v1 = __riscv_vabsdiff<vint32m1_t, vuint32m1_t>(v11, v12);
        r1 = __riscv_vmaxu(r1, v1, vle32m1);
    }
    return (int)__riscv_vmv_x(__riscv_vredmaxu(__riscv_vmaxu(r0, r1, vle32m1), r0, vle32m1));
}

template <typename T, typename ST> inline
ST normDiffL1_rvv(const T* src1, const T* src2, int n, int& j);

template<> inline
int normDiffL1_rvv(const schar* src1, const schar* src2, int n, int& j) {
    const int vle8m1 = __riscv_vsetvlmax_e8m1();
    const int vle16m1 = __riscv_vsetvlmax_e16m1();
    const int vle32m1 = __riscv_vsetvlmax_e32m1();
    vuint32m1_t r0 = __riscv_vmv_v_x_u32m1(0, vle32m1);
    vuint32m1_t r1 = __riscv_vmv_v_x_u32m1(0, vle32m1);
    vuint16m1_t zero = __riscv_vmv_v_x_u16m1(0, vle16m1);
    for (; j <= n - 2 * vle8m1; j += 2 * vle8m1) {
        vint8m1_t v01 = __riscv_vle8_v_i8m1(src1 + j, vle8m1), v02 = __riscv_vle8_v_i8m1(src2 + j, vle8m1);
        vuint8m1_t v0 = __riscv_vabsdiff<vint8m1_t, vuint8m1_t>(v01, v02);
        vuint16m1_t u0 = __riscv_vwredsumu_tu(zero, v0, zero, vle8m1);
        r0 = __riscv_vwredsumu(u0, r0, vle16m1);

        vint8m1_t v11 = __riscv_vle8_v_i8m1(src1 + j + vle8m1, vle8m1),
                  v12 = __riscv_vle8_v_i8m1(src2 + j + vle8m1, vle8m1);
        vuint8m1_t v1 = __riscv_vabsdiff<vint8m1_t, vuint8m1_t>(v11, v12);
        vuint16m1_t u1 = __riscv_vwredsumu_tu(zero, v1, zero, vle8m1);
        r1 = __riscv_vwredsumu(u1, r1, vle16m1);
    }
    return (int)__riscv_vmv_x(__riscv_vadd(r0, r1, vle32m1));
}

template<> inline
int normDiffL1_rvv(const ushort* src1, const ushort* src2, int n, int& j) {
    const int vle16m1 = __riscv_vsetvlmax_e16m1();
    const int vle32m1 = __riscv_vsetvlmax_e32m1();
    vuint32m1_t r0 = __riscv_vmv_v_x_u32m1(0, vle32m1);
    vuint32m1_t r1 = __riscv_vmv_v_x_u32m1(0, vle32m1);
    for (; j <= n - 2 * vle16m1; j += 2 * vle16m1) {
        vuint16m1_t v01 = __riscv_vle16_v_u16m1(src1 + j, vle16m1),
                    v02 = __riscv_vle16_v_u16m1(src2 + j, vle16m1);
        vuint16m1_t v0 = __riscv_vabsdiff<vuint16m1_t, vuint16m1_t>(v01, v02);
        r0 = __riscv_vwredsumu(v0, r0, vle16m1);

        vuint16m1_t v11 = __riscv_vle16_v_u16m1(src1 + j + vle16m1, vle16m1),
                    v12 = __riscv_vle16_v_u16m1(src2 + j + vle16m1, vle16m1);
        vuint16m1_t v1 = __riscv_vabsdiff<vuint16m1_t, vuint16m1_t>(v11, v12);
        r1 = __riscv_vwredsumu(v1, r1, vle16m1);
    }
    return (int)__riscv_vmv_x(__riscv_vadd(r0, r1, vle32m1));
}

template <typename T, typename ST> inline
ST normDiffL2_rvv(const T* src1, const T* src2, int n, int& j);

template<> inline
int normDiffL2_rvv(const schar* src1, const schar* src2, int n, int& j) {
    const int vle8m1 = __riscv_vsetvlmax_e8m1();
    vuint32m4_t r0 = __riscv_vmv_v_x_u32m4(0, vle8m1);
    vuint32m4_t r1 = __riscv_vmv_v_x_u32m4(0, vle8m1);
    for (; j <= n - 2 * vle8m1; j += 2 * vle8m1) {
        vint8m1_t v01 = __riscv_vle8_v_i8m1(src1 + j, vle8m1),
                  v02 = __riscv_vle8_v_i8m1(src2 + j, vle8m1);
        vuint8m1_t v0 = __riscv_vabsdiff<vint8m1_t, vuint8m1_t>(v01, v02);
        vuint16m2_t u0 = __riscv_vwmulu(v0, v0, vle8m1);
        r0 = __riscv_vadd(r0, __riscv_vwcvtu_x(u0, vle8m1), vle8m1);

        vint8m1_t v11 = __riscv_vle8_v_i8m1(src1 + j + vle8m1, vle8m1),
                  v12 = __riscv_vle8_v_i8m1(src2 + j + vle8m1, vle8m1);
        vuint8m1_t v1 = __riscv_vabsdiff<vint8m1_t, vuint8m1_t>(v11, v12);
        vuint16m2_t u1 = __riscv_vwmulu(v1, v1, vle8m1);
        r1 = __riscv_vadd(r1, __riscv_vwcvtu_x(u1, vle8m1), vle8m1);
    }
    return (int)__riscv_vmv_x(
                __riscv_vwredsumu(__riscv_vadd(r0, r1, vle8m1),
                                  __riscv_vmv_v_x_u64m1(0, __riscv_vsetvlmax_e64m1()),
                                  vle8m1)
            );
}

template<> inline
double normDiffL2_rvv(const ushort* src1, const ushort* src2, int n, int& j) {
    const int vle16m1 = __riscv_vsetvlmax_e16m1();
    const int vle32m2 = __riscv_vsetvlmax_e32m2();
    const int vle64m4 = __riscv_vsetvlmax_e64m4();
    vfloat64m4_t r0 = __riscv_vfmv_v_f_f64m4(0, vle64m4);
    vfloat64m4_t r1 = __riscv_vfmv_v_f_f64m4(0, vle64m4);
    for (; j <= n - 2 * vle16m1; j += 2 * vle16m1) {
        vuint16m1_t v01 = __riscv_vle16_v_u16m1(src1 + j, vle16m1),
                    v02 = __riscv_vle16_v_u16m1(src2 + j, vle16m1);
        vuint16m1_t v0 = __riscv_vabsdiff<vuint16m1_t, vuint16m1_t>(v01, v02);
        vuint32m2_t u0 = __riscv_vwmulu(v0, v0, vle16m1);
        r0 = __riscv_vfadd(r0, __riscv_vfwcvt_f(u0, vle32m2), vle64m4);

        vuint16m1_t v11 = __riscv_vle16_v_u16m1(src1 + j + vle16m1, vle16m1),
                    v12 = __riscv_vle16_v_u16m1(src2 + j + vle16m1, vle16m1);
        vuint16m1_t v1 = __riscv_vabsdiff<vuint16m1_t, vuint16m1_t>(v11, v12);
        vuint32m2_t u1 = __riscv_vwmulu(v1, v1, vle16m1);
        r1 = __riscv_vfadd(r1, __riscv_vfwcvt_f(u1, vle32m2), vle64m4);
    }
    return __riscv_vfmv_f(
        __riscv_vfredosum(
            __riscv_vfadd(r0, r1, vle64m4),
            __riscv_vfmv_v_f_f64m1(0, __riscv_vsetvlmax_e64m1()),
            vle64m4)
    );
}

CV_CPU_OPTIMIZATION_NAMESPACE_END

} // cv::
