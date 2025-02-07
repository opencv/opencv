#ifndef OPENCV_HAL_RVV_EXP64F_HPP_INCLUDED
#define OPENCV_HAL_RVV_EXP64F_HPP_INCLUDED

#include <opencv2/core/cvdef.h>

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_exp64f
#define cv_hal_exp64f cv::cv_hal_rvv::exp64f
#define CL_M_LOG2E 1.442695040888963407359924681001892137

static const long double factor_term [10] {1.0/2.0, 1.0/6.0, 1.0/24.0, 1.0/120.0, 1.0/720.0, 1.0/5040.0, 1.0/40320.0, 1.0/362880.0, 1.0/3628800.0, 1.0/39916800.0};

inline vfloat64m8_t pow2_v_i64m8_f64m8(vint64m8_t n, size_t vl)
{
    const int64_t BIAS_64 = 1023;
    vint64m8_t exp = __riscv_vadd_vx_i64m8(n, BIAS_64, vl);
    vint64m8_t bits = __riscv_vsll_vx_i64m8(exp, 52, vl);
    return __riscv_vreinterpret_v_i64m8_f64m8(bits);
}

inline int exp64f(const double* src, double* dst, int n) {
    if (n <= 0) return 0;

    ssize_t i = 0;
    ssize_t vlmax = __riscv_vsetvlmax_e64m8();

    while (i < n) {
        ssize_t vl = n - i;
        if (vl > vlmax) vl = vlmax;
        vl = __riscv_vsetvl_e64m8(vl);

        vfloat64m8_t vx = __riscv_vle64_v_f64m8(src + i, vl);
        vx = __riscv_vfmul_vf_f64m8(vx, CL_M_LOG2E, vl);

        vint64m8_t int_part = __riscv_vfcvt_x_f_v_i64m8(vx, vl);
        vfloat64m8_t vpow2n = pow2_v_i64m8_f64m8(int_part, vl);

        vfloat64m8_t frac_part = __riscv_vfsub_vv_f64m8(vx, 
                                __riscv_vfcvt_f_x_v_f64m8(int_part, vl), vl);
        frac_part = __riscv_vfmul_vf_f64m8(frac_part, CV_LOG2, vl);

        vfloat64m8_t term = frac_part;
        vfloat64m8_t res = __riscv_vfadd_vf_f64m8(frac_part, 1.0, vl);

        for (int j = 0; j < 10; ++j) {
            term = __riscv_vfmul_vv_f64m8(term, frac_part, vl);
            res = __riscv_vfmacc_vf_f64m8(res, factor_term[j], term, vl);
        }
    
        res = __riscv_vfmul_vv_f64m8(res, vpow2n, vl);
        __riscv_vse64_v_f64m8(dst + i, res, vl);

        i += vl;
    }

    return 0;
}

} }

#endif
