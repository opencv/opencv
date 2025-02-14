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
int normL2_rvv(const schar* src, int n, int& j) {
    const int vle8m1 = __riscv_vsetvlmax_e8m1();
    const int vle16m1 = __riscv_vsetvlmax_e16m1();
    const int vle32m1 = __riscv_vsetvlmax_e32m1();
    const int vle64m1 = __riscv_vsetvlmax_e64m1();
    vint32m1_t r0 = __riscv_vmv_v_x_i32m1(0, vle32m1), r1 = __riscv_vmv_v_x_i32m1(0, vle32m1);
    for (; j <= n - 2 * vle8m1; j += 2 * vle8m1) {
        vint8m1_t v0 = __riscv_vle8_v_i8m1(src + j, vle8m1);
        vint16m2_t s0 = __riscv_vwmul_vv_i16m2(v0, v0, vle8m1);
        r0 = __riscv_vwredsum_vs_i16m2_i32m1(s0, r0, vle16m1);

        vint8m1_t v1 = __riscv_vle8_v_i8m1(src + j + vle8m1, vle8m1);
        vint16m2_t s1 = __riscv_vwmul_vv_i16m2(v1, v1, vle8m1);
        r1 = __riscv_vwredsum_vs_i16m2_i32m1(s1, r1, vle16m1);
    }
    r0 = __riscv_vsadd(r0, r1, vle32m1);
    vint64m1_t zero = __riscv_vmv_v_x_i64m1(0, vle64m1);
    vint64m1_t res = __riscv_vwredsum(r0, zero, vle32m1);
    return (int)__riscv_vmv_x(res);
}

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
