// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#ifndef OPENCV_HAL_RVV_INTEGRAL_HPP_INCLUDED
#define OPENCV_HAL_RVV_INTEGRAL_HPP_INCLUDED

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_integral
#define cv_hal_integral cv::cv_hal_rvv::integral

template <typename T> struct rvv;

// Vector operations wrapper
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
    static inline vint32m4_t vmul(vint32m4_t a, vint32m4_t b, size_t vl) { return __riscv_vmul_vv_i32m4(a, b, vl); }
    static inline vint32m4_t permute(vint32m4_t a, int cn, size_t vl) {
        auto v_last = __riscv_vslidedown_vx_i32m4(a, vl - cn, vl);
        if (cn == 1) return __riscv_vmv_v_x_i32m4(__riscv_vmv_x_s_i32m4_i32(v_last), vl);
        for (size_t offset = cn; offset < vl; offset <<= 1) {
            v_last = __riscv_vslideup_vx_i32m4(v_last, v_last, offset, vl);
        }
        return v_last;
    }
};

template <> struct rvv<uint16_t> {
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e16m2(a); }
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e16m2(); }
    static inline vuint16m2_t vle(const uint16_t* a, size_t vl) { return __riscv_vle16_v_u16m2(a, vl); }
    static inline void vse(uint16_t* a, vuint16m2_t b, size_t vl) { __riscv_vse16_v_u16m2(a, b, vl); }
    
    static inline vuint16m2_t vmv(uint16_t a, size_t vl) { return __riscv_vmv_v_x_u16m2(a, vl); }
    static inline vuint16m2_t vslideup(vuint16m2_t a, vuint16m2_t b, size_t n, size_t vl) { return __riscv_vslideup_vx_u16m2(a, b, n, vl); }
    static inline vuint16m2_t vadd(vuint16m2_t a, vuint16m2_t b, size_t vl) { return __riscv_vadd_vv_u16m2(a, b, vl); }
    static inline vuint16m2_t vadd(vuint16m2_t a, uint16_t b, size_t vl) { return __riscv_vadd_vx_u16m2(a, b, vl); }
    static inline vuint16m2_t vmul(vuint16m2_t a, vuint16m2_t b, size_t vl) { return __riscv_vmul_vv_u16m2(a, b, vl); }
    static inline vuint16m2_t permute(vuint16m2_t a, int cn, size_t vl) {
        auto v_last = __riscv_vslidedown_vx_u16m2(a, vl - cn, vl);
        if (cn == 1) return __riscv_vmv_v_x_u16m2(__riscv_vmv_x_s_u16m2_u16(v_last), vl);
        for (size_t offset = cn; offset < vl; offset <<= 1) {
            v_last = __riscv_vslideup_vx_u16m2(v_last, v_last, offset, vl);
        }
        return v_last;
    }
};

template <> struct rvv<int16_t> {
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e16m2(a); }
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e16m2(); }
    static inline vint16m2_t vle(const int16_t* a, size_t vl) { return __riscv_vle16_v_i16m2(a, vl); }
    static inline void vse(int16_t* a, vint16m2_t b, size_t vl) { __riscv_vse16_v_i16m2(a, b, vl); }
    
    static inline vint16m2_t vmv(int16_t a, size_t vl) { return __riscv_vmv_v_x_i16m2(a, vl); }
    static inline vint16m2_t vslideup(vint16m2_t a, vint16m2_t b, size_t n, size_t vl) { return __riscv_vslideup_vx_i16m2(a, b, n, vl); }
    static inline vint16m2_t vadd(vint16m2_t a, vint16m2_t b, size_t vl) { return __riscv_vadd_vv_i16m2(a, b, vl); }
    static inline vint16m2_t vadd(vint16m2_t a, int16_t b, size_t vl) { return __riscv_vadd_vx_i16m2(a, b, vl); }
    static inline vint16m2_t vmul(vint16m2_t a, vint16m2_t b, size_t vl) { return __riscv_vmul_vv_i16m2(a, b, vl); }
    static inline vint16m2_t permute(vint16m2_t a, int cn, size_t vl) {
        auto v_last = __riscv_vslidedown_vx_i16m2(a, vl - cn, vl);
        if (cn == 1) return __riscv_vmv_v_x_i16m2(__riscv_vmv_x_s_i16m2_i16(v_last), vl);
        for (size_t offset = cn; offset < vl; offset <<= 1) {
            v_last = __riscv_vslideup_vx_i16m2(v_last, v_last, offset, vl);
        }
        return v_last;
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
    static inline vfloat32m4_t vmul(vfloat32m4_t a, vfloat32m4_t b, size_t vl) { return __riscv_vfmul_vv_f32m4(a, b, vl); }
    static inline vfloat32m4_t permute(vfloat32m4_t a, int cn, size_t vl) {
        auto v_last = __riscv_vslidedown_vx_f32m4(a, vl - cn, vl);
        if (cn == 1) return __riscv_vfmv_v_f_f32m4(__riscv_vfmv_f_s_f32m4_f32(v_last), vl);
        for (size_t offset = cn; offset < vl; offset <<= 1) {
            v_last = __riscv_vslideup_vx_f32m4(v_last, v_last, offset, vl);
        }
        return v_last;
    }
};

template <> struct rvv<double> {
    static inline size_t vsetvl(size_t a) { return __riscv_vsetvl_e64m4(a); }
    static inline size_t vsetvlmax() { return __riscv_vsetvlmax_e64m4(); }
    static inline vfloat64m4_t vle(const double* a, size_t vl) { return __riscv_vle64_v_f64m4(a, vl); }
    static inline void vse(double* a, vfloat64m4_t b, size_t vl) { __riscv_vse64_v_f64m4(a, b, vl); }

    static inline vfloat64m4_t vmv(double a, size_t vl) { return __riscv_vfmv_v_f_f64m4(a, vl); }
    static inline vfloat64m4_t vslideup(vfloat64m4_t a, vfloat64m4_t b, size_t n, size_t vl) { return __riscv_vslideup_vx_f64m4(a, b, n, vl); }
    static inline vfloat64m4_t vadd(vfloat64m4_t a, vfloat64m4_t b, size_t vl) { return __riscv_vfadd_vv_f64m4(a, b, vl); }
    static inline vfloat64m4_t vadd(vfloat64m4_t a, double b, size_t vl) { return __riscv_vfadd_vf_f64m4(a, b, vl); }
    static inline vfloat64m4_t vmul(vfloat64m4_t a, vfloat64m4_t b, size_t vl) { return __riscv_vfmul_vv_f64m4(a, b, vl); }
    static inline vfloat64m4_t permute(vfloat64m4_t a, int cn, size_t vl) {
        auto v_last = __riscv_vslidedown_vx_f64m4(a, vl - cn, vl);
        if (cn == 1) return __riscv_vfmv_v_f_f64m4(__riscv_vfmv_f_s_f64m4_f64(v_last), vl);
        for (size_t offset = cn; offset < vl; offset <<= 1) {
            v_last = __riscv_vslideup_vx_f64m4(v_last, v_last, offset, vl);
        }
        return v_last;
    }
};

// Conversion between data type and accumulator type
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

template<> struct vec_cvt<uint8_t, double> {
    static inline vfloat64m4_t convert(vuint8m1_t a, size_t vl) {
        vuint64m8_t v_u64 = __riscv_vzext_vf8_u64m8(a, vl);
        return __riscv_vget_v_f64m8_f64m4(__riscv_vfcvt_f_xu_v_f64m8(v_u64, vl), 0);
    }
};

template<> struct vec_cvt<float, float> {
    static inline vfloat32m4_t convert(vfloat32m4_t a, [[maybe_unused]] size_t vl) { return a; }
};

template<> struct vec_cvt<float, double> {
    static inline vfloat64m4_t convert(vfloat32m4_t a, size_t vl) {
        return __riscv_vget_v_f64m8_f64m4( __riscv_vfwcvt_f_f_v_f64m8(a, vl), 0);
    }
};

template<> struct vec_cvt<double, double> {
    static inline vfloat64m4_t convert(vfloat64m4_t a, [[maybe_unused]] size_t vl) { return a; }
};

template<> struct vec_cvt<uint16_t, double> {
    static inline vfloat64m4_t convert(vuint16m2_t a, size_t vl) {
        vuint64m8_t v_u64 = __riscv_vzext_vf4_u64m8(a, vl);
        return __riscv_vget_v_f64m8_f64m4(__riscv_vfcvt_f_xu_v_f64m8(v_u64, vl), 0);
    }
};

template<> struct vec_cvt<int16_t, double> {
    static inline vfloat64m4_t convert(vint16m2_t a, size_t vl) {
        vint64m8_t v_i64 = __riscv_vsext_vf4_i64m8(a, vl);
        return __riscv_vget_v_f64m8_f64m4(__riscv_vfcvt_f_x_v_f64m8(v_i64, vl), 0);
    }
};


template <typename data_t, typename acc_t, bool sqsum = false>
inline int integral_inner(const uchar* src_data, size_t src_step,
                          uchar* sum_data, size_t sum_step,
                          int width, int height, int cn) {
    for (int y = 0; y < height; y++) {
        const data_t* src = reinterpret_cast<const data_t*>(src_data + src_step * y);
        acc_t* prev = reinterpret_cast<acc_t*>(sum_data + sum_step * y);
        acc_t* curr = reinterpret_cast<acc_t*>(sum_data + sum_step * (y + 1));
        memset(curr, 0, cn * sizeof(acc_t));

        size_t vl = rvv<acc_t>::vsetvlmax();
        auto sum = rvv<acc_t>::vmv(0, vl);
        for (size_t x = 0; x < static_cast<size_t>(width); x += vl) {
            vl = rvv<acc_t>::vsetvl(width - x);
            __builtin_prefetch(&src[x + vl], 0);
            __builtin_prefetch(&prev[x + cn], 0);

            auto v_src = rvv<data_t>::vle(&src[x], vl);
            auto v_zero = rvv<acc_t>::vmv(0, vl);
            auto acc = vec_cvt<data_t, acc_t>::convert(v_src, vl);

            if (sqsum) { // Squared Sum
                acc = rvv<acc_t>::vmul(acc, acc, vl);
            }

            for (size_t offset = cn; offset < vl; offset <<= 1) {
                auto v_shift = rvv<acc_t>::vslideup(v_zero, acc, offset, vl);
                acc = rvv<acc_t>::vadd(acc, v_shift, vl);
            }
            // Extract last cn elements from vector, which is needed for the next iteration
            // and repeat them for vlmax / cn times
            auto last_n = rvv<acc_t>::permute(acc, cn, vl);

            auto v_prev = rvv<acc_t>::vle(&prev[x + cn], vl);
            acc = rvv<acc_t>::vadd(acc, v_prev, vl);
            acc = rvv<acc_t>::vadd(acc, sum, vl);
            sum = rvv<acc_t>::vadd(sum, last_n, vl);

            rvv<acc_t>::vse(&curr[x + cn], acc, vl);
        }
    }

    return CV_HAL_ERROR_OK;
}

template <typename data_t, typename acc_t, typename sq_acc_t>
inline int integral(const uchar* src_data, size_t src_step, uchar* sum_data, size_t sum_step, uchar* sqsum_data, size_t sqsum_step, int width, int height, int cn) {
    memset(sum_data, 0, (sum_step) * sizeof(uchar));

    int result = CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (sqsum_data == nullptr) {
        result = integral_inner<data_t, acc_t, false>(src_data, src_step, sum_data, sum_step, width, height, cn);
    } else {
        result = integral_inner<data_t, acc_t, false>(src_data, src_step, sum_data, sum_step, width, height, cn);
        memset(sqsum_data, 0, (sqsum_step) * sizeof(uchar));
        result = integral_inner<data_t, sq_acc_t, true>(src_data, src_step, sqsum_data, sqsum_step, width, height, cn);
    }
    return result;
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
   @note Following combinations of image depths are used:
    Source | Sum | Square sum
    -------|-----|-----------
    CV_8U | CV_32S | CV_64F
    CV_8U | CV_32S | CV_32F
    CV_8U | CV_32S | CV_32S
    CV_8U | CV_32F | CV_64F
    CV_8U | CV_32F | CV_32F
    CV_8U | CV_64F | CV_64F
    CV_16U | CV_64F | CV_64F
    CV_16S | CV_64F | CV_64F
    CV_32F | CV_32F | CV_64F
    CV_32F | CV_32F | CV_32F
    CV_32F | CV_64F | CV_64F
    CV_64F | CV_64F | CV_64F
*/
inline int integral(int depth, int sdepth, int sqdepth,
                    const uchar* src_data, size_t src_step,
                    uchar* sum_data, size_t sum_step,
                    uchar* sqsum_data, size_t sqsum_step,
                    uchar* tilted_data, [[maybe_unused]] size_t tilted_step,
                    int width, int height, int cn) {
    // tilted sum and cn == 3 cases are not supported
    if (tilted_data || cn == 3) {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    // Skip images that are too small
    if (!(width >> 8 || height >> 8)) {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    int result = CV_HAL_ERROR_NOT_IMPLEMENTED;

    width *= cn;

    if( depth == CV_8U && sdepth == CV_32S && sqdepth == CV_64F )
        result = integral<uchar, int, double>(src_data, src_step, sum_data, sum_step, sqsum_data, sqsum_step, width, height, cn);
    else if( depth == CV_8U && sdepth == CV_32S && sqdepth == CV_32F )
        result = integral<uchar, int, float>(src_data, src_step, sum_data, sum_step, sqsum_data, sqsum_step, width, height, cn);
    else if( depth == CV_8U && sdepth == CV_32S && sqdepth == CV_32S )
        result = integral<uchar, int, int>(src_data, src_step, sum_data, sum_step, sqsum_data, sqsum_step, width, height, cn);
    else if( depth == CV_8U && sdepth == CV_32F && sqdepth == CV_64F )
        result = integral<uchar, float, double>(src_data, src_step, sum_data, sum_step, sqsum_data, sqsum_step, width, height, cn);
    else if( depth == CV_8U && sdepth == CV_32F && sqdepth == CV_32F )
        result = integral<uchar, float, float>(src_data, src_step, sum_data, sum_step, sqsum_data, sqsum_step, width, height, cn);
    else if( depth == CV_8U && sdepth == CV_64F && sqdepth == CV_64F )
        result = integral<uchar, double, double>(src_data, src_step, sum_data, sum_step, sqsum_data, sqsum_step, width, height, cn);
    else if( depth == CV_16U && sdepth == CV_64F && sqdepth == CV_64F )
        result = integral<ushort, double, double>(src_data, src_step, sum_data, sum_step, sqsum_data, sqsum_step, width, height, cn);
    else if( depth == CV_16S && sdepth == CV_64F && sqdepth == CV_64F )
        result = integral<short, double, double>(src_data, src_step, sum_data, sum_step, sqsum_data, sqsum_step, width, height, cn);
    else if( depth == CV_32F && sdepth == CV_32F && sqdepth == CV_64F )
        result = integral<float, float, double>(src_data, src_step, sum_data, sum_step, sqsum_data, sqsum_step, width, height, cn);
    else if( depth == CV_32F && sdepth == CV_32F && sqdepth == CV_32F )
        result = integral<float, float, float>(src_data, src_step, sum_data, sum_step, sqsum_data, sqsum_step, width, height, cn);
    else if( depth == CV_32F && sdepth == CV_64F && sqdepth == CV_64F )
        result = integral<float, double, double>(src_data, src_step, sum_data, sum_step, sqsum_data, sqsum_step, width, height, cn);
    else if( depth == CV_64F && sdepth == CV_64F && sqdepth == CV_64F ) {
        result = integral<double, double, double>(src_data, src_step, sum_data, sum_step, sqsum_data, sqsum_step, width, height, cn);
    }

    return result;
}

}}

#endif
