// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_INTEGRAL_HPP_INCLUDED
#define OPENCV_HAL_RVV_INTEGRAL_HPP_INCLUDED

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_integral
#define cv_hal_integral cv::cv_hal_rvv::integral

template <typename T> struct rvv;

template<> struct rvv<uint8_t> {
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e8m1(a); }
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e8m1(); }
    static inline vuint8m1_t vle(const uint8_t* a, size_t vl) { return __riscv_vle8_v_u8m1(a, vl); }
    static inline void vse(uint8_t* a, vuint8m1_t b, size_t vl) { __riscv_vse8_v_u8m1(a, b, vl); }
};

template <> struct rvv<int> {
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e32m4(a); }
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e32m4(); }
    static inline vint32m4_t vle(const int* a, size_t vl) { return __riscv_vle32_v_i32m4(a, vl); }
    static inline void vse(int* a, vint32m4_t b, size_t vl) { __riscv_vse32_v_i32m4(a, b, vl); }

    static inline vint32m4_t vmv(int a, size_t vl) { return __riscv_vmv_v_x_i32m4(a, vl); }
    static inline vint32m4_t vslideup(vint32m4_t a, vint32m4_t b, size_t n, size_t vl) { return __riscv_vslideup_vx_i32m4(a, b, n, vl); }
    static inline vint32m4_t vadd(vint32m4_t a, vint32m4_t b, size_t vl) { return __riscv_vadd_vv_i32m4(a, b, vl); }
    static inline vint32m4_t vadd(vint32m4_t a, int b, size_t vl) { return __riscv_vadd_vx_i32m4(a, b, vl); }
    static inline int last(vint32m4_t a, size_t vl) {
        return __riscv_vmv_x_s_i32m4_i32(__riscv_vslidedown_vx_i32m4(a, vl - 1, vl));
    }
};

template <> struct rvv<float> {
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e32m4(a); }
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e32m4(); }
    static inline vfloat32m4_t vle(const float* a, size_t vl) { return __riscv_vle32_v_f32m4(a, vl); }
    static inline void vse(float* a, vfloat32m4_t b, size_t vl) { __riscv_vse32_v_f32m4(a, b, vl); }

    static inline vfloat32m4_t vmv(float a, size_t vl) { return __riscv_vfmv_v_f_f32m4(a, vl); }
    static inline vfloat32m4_t vslideup(vfloat32m4_t a, vfloat32m4_t b, size_t n, size_t vl) { return __riscv_vslideup_vx_f32m4(a, b, n, vl); }
    static inline vfloat32m4_t vadd(vfloat32m4_t a, vfloat32m4_t b, size_t vl) { return __riscv_vfadd_vv_f32m4(a, b, vl); }
    static inline vfloat32m4_t vadd(vfloat32m4_t a, float b, size_t vl) { return __riscv_vfadd_vf_f32m4(a, b, vl); }
    static inline float last(vfloat32m4_t a, size_t vl) {
        return __riscv_vfmv_f_s_f32m4_f32(__riscv_vslidedown_vx_f32m4(a, vl - 1, vl));
    }
};

template <> struct rvv<double> {
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e64m2(a); }
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e64m2(); }
    static inline vfloat64m2_t vle(const double* a, size_t vl) { return __riscv_vle64_v_f64m2(a, vl); }
    static inline void vse(double* a, vfloat64m2_t b, size_t vl) { __riscv_vse64_v_f64m2(a, b, vl); }

    static inline vfloat64m2_t vmv(double a, size_t vl) { return __riscv_vfmv_v_f_f64m2(a, vl); }
    static inline vfloat64m2_t vslideup(vfloat64m2_t a, vfloat64m2_t b, size_t n, size_t vl) { return __riscv_vslideup_vx_f64m2(a, b, n, vl); }
    static inline vfloat64m2_t vadd(vfloat64m2_t a, vfloat64m2_t b, size_t vl) { return __riscv_vfadd_vv_f64m2(a, b, vl); }
    static inline vfloat64m2_t vadd(vfloat64m2_t a, double b, size_t vl) { return __riscv_vfadd_vf_f64m2(a, b, vl); }
    static inline double last(vfloat64m2_t a, size_t vl) {
        return __riscv_vfmv_f_s_f64m2_f64(__riscv_vslidedown_vx_f64m2(a, vl - 1, vl));
    }
};

template <typename data_t, typename acc_t> struct vec_cvt;
template<> struct vec_cvt<uint8_t, int> {
    static inline vint32m4_t convert(vuint8m1_t a, size_t vl) {
        vuint32m4_t v_u32 = __riscv_vzext_vf4_u32m4(a, vl);
        return __riscv_vreinterpret_v_u32m4_i32m4(v_u32);
    }
};

template<> struct vec_cvt<uint8_t, float> {
    static inline vfloat32m4_t convert(vuint8m1_t a, size_t vl) {
        vuint32m4_t v_u32 = __riscv_vzext_vf4_u32m4(a, vl);
        return 	__riscv_vfcvt_f_xu_v_f32m4(v_u32, vl);
    }
};

template<> struct vec_cvt<float, float> {
    static inline vfloat32m4_t convert(vfloat32m4_t a, [[maybe_unused]] size_t vl) { return a; }
};

template <typename data_t, typename acc_t>
inline int integral(const uchar * src_data, size_t src_step, uchar * sum_data, size_t sum_step,  int width, int height) {
    memset(sum_data, 0, (sum_step) * sizeof(uchar));

    for (int y = 0; y < height; y++) {
        const data_t* src = reinterpret_cast<const data_t*>(src_data + src_step * y);
        acc_t* prev = reinterpret_cast<acc_t*>(sum_data + sum_step * y);
        acc_t* curr = reinterpret_cast<acc_t*>(sum_data + sum_step * (y + 1));
        curr[0] = 0;

        int vl = 0;
        acc_t sum = 0, last_sum = 0;

        for (int x = 0; x < width; x += vl) {
            vl = rvv<data_t>::vsetvl(width - x);
            auto v_src = rvv<data_t>::vle(&src[x], vl);
            auto v_prev = rvv<acc_t>::vle(&prev[x + 1], vl);
            auto v_zero = rvv<acc_t>::vmv(0, vl);
            auto acc = vec_cvt<data_t, acc_t>::convert(v_src, vl);

            for (int offset = 1; offset < vl; offset <<= 1) {
                auto v_shift = rvv<acc_t>::vslideup(v_zero, acc, offset, vl);
                acc = rvv<acc_t>::vadd(acc, v_shift, vl);
            }
            last_sum = rvv<acc_t>::last(acc, vl);

            acc = rvv<acc_t>::vadd(acc, v_prev, vl);
            acc = rvv<acc_t>::vadd(acc, sum, vl);
            sum += last_sum;

            rvv<acc_t>::vse(&curr[x + 1], acc, vl);
        }
    }

    return CV_HAL_ERROR_OK;
}

/**
   @brief Calculate integral image
   @param depth Depth of source image
   @param sdepth Depth of sum image
   @param sqdepth Depth of square sum image
   @param src_data Source image data
   @param src_step Source image step
   @param sum_data Sum image data
   @param sum_step Sum image step
   @param sqsum_data Square sum image data
   @param sqsum_step Square sum image step
   @param tilted_data Tilted sum image data
   @param tilted_step Tilted sum image step
   @param width Source image width
   @param height Source image height
   @param cn Number of channels
*/
inline int integral(int depth, int sdepth, [[maybe_unused]] int sqdepth,
                    const uchar * src_data, size_t src_step,
                    uchar * sum_data, size_t sum_step,
                    uchar * sqsum_data, [[maybe_unused]] size_t sqsum_step,
                    uchar * tilted_data, [[maybe_unused]] size_t tilted_step,
                    int width, int height, int cn) {
    if (sqsum_data || tilted_data || cn > 1)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    int result = CV_HAL_ERROR_NOT_IMPLEMENTED;

    if (depth == CV_8U && sdepth == CV_32S) {
        result = integral<uint8_t, int>(src_data, src_step, sum_data, sum_step, width, height);
    } else if (depth == CV_8U && sdepth == CV_32F) {
        result = integral<uint8_t, float>(src_data, src_step, sum_data, sum_step, width, height);
    } else if (depth == CV_32F && sdepth == CV_32F) {
        result = integral<float, float>(src_data, src_step, sum_data, sum_step, width, height);
    }

    return result;
}

}}

#endif
