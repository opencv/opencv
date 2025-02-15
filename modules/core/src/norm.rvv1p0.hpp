// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copytright (C) 2025, SpaceMIT Inc., all rights reserved.

#include "opencv2/core/hal/intrin.hpp"

namespace cv {

CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

template <typename T, typename ST> inline
ST normL1_rvv(const T* src, int n, int& j);

template<> inline
int normL1_rvv(const ushort* src, int n, int& j) {
    const int vle16m1 = __riscv_vsetvlmax_e16m1();
    const int vle32m1 = __riscv_vsetvlmax_e32m1();
    vuint32m1_t r = __riscv_vmv_v_x_u32m1(0, vle32m1);
    for (; j <= n - vle16m1; j += vle16m1) {
        vuint16m1_t v = __riscv_vle16_v_u16m1(src + j, vle16m1);
        r = __riscv_vwredsumu(v, r, vle16m1);
    }
    return (int)__riscv_vmv_x(r);
}

template<> inline
int normL1_rvv(const short* src, int n, int& j) {
    const int vle16m1 = __riscv_vsetvlmax_e16m1();
    const int vle32m1 = __riscv_vsetvlmax_e32m1();
    vuint32m1_t r = __riscv_vmv_v_x_u32m1(0, vle32m1);
    for (; j<= n - vle16m1; j += vle16m1) {
        vint16m1_t v = __riscv_vle16_v_i16m1(src + j, vle16m1);
        vint16m1_t mask = __riscv_vsra_vx_i16m1(v, 15, vle16m1);
        vint16m1_t v_xor = __riscv_vxor_vv_i16m1(v, mask, vle16m1);
        vuint16m1_t v_abs = __riscv_vreinterpret_v_i16m1_u16m1(__riscv_vsub_vv_i16m1(v_xor, mask, vle16m1));
        r = __riscv_vwredsumu(v_abs, r, vle16m1);
    }
    return (int)__riscv_vmv_x(r);
}

template<> inline
double normL1_rvv(const double* src, int n, int& j) {
    const int vle64m1 = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t r00 = __riscv_vfmv_v_f_f64m1(0.f, vle64m1);
    for (; j <= n - vle64m1; j += vle64m1) {
        vfloat64m1_t v00 = __riscv_vle64_v_f64m1(src + j, vle64m1);
        v00 = __riscv_vfabs(v00, vle64m1);
        r00 = __riscv_vfadd(r00, v00, vle64m1);
    }
    vfloat64m1_t s00 = __riscv_vfmv_v_f_f64m1(0.f, vle64m1);
    s00 = __riscv_vfredusum(r00, __riscv_vfmv_v_f_f64m1(0.f, vle64m1), vle64m1);
    return __riscv_vfmv_f(s00);
}

template <typename T, typename ST> inline
ST normL2_rvv(const T* src, int n, int& j);

template<> inline
double normL2_rvv(const double* src, int n, int& j) {
    const int vle64m1 = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t r00 = __riscv_vfmv_v_f_f64m1(0.f, vle64m1);
    for (; j <= n - vle64m1; j += vle64m1) {
        vfloat64m1_t v00 = __riscv_vle64_v_f64m1(src + j, vle64m1);
        r00 = __riscv_vfmacc(r00, v00, v00, vle64m1);
    }
    vfloat64m1_t s00 = __riscv_vfmv_v_f_f64m1(0.f, vle64m1);
    s00 = __riscv_vfredusum(r00, __riscv_vfmv_v_f_f64m1(0.f, vle64m1), vle64m1);
    return __riscv_vfmv_f(s00);
}

CV_CPU_OPTIMIZATION_NAMESPACE_END

} // cv::
