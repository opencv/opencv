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
