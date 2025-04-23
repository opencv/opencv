// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#ifndef OPENCV_HAL_RVV_INTEGRAL_HPP_INCLUDED
#define OPENCV_HAL_RVV_INTEGRAL_HPP_INCLUDED

#include <riscv_vector.h>
#include "types.hpp"

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_integral
#define cv_hal_integral cv::cv_hal_rvv::integral

template <typename vec_t>
inline typename vec_t::VecType repeat_last_n(typename vec_t::VecType vs, int n, size_t vl) {
    auto v_last = vec_t::vslidedown(vs, vl - n, vl);
    if (n == 1) return vec_t::vmv(vec_t::vmv_x(v_last), vl);
    for (size_t offset = n; offset < vl; offset <<= 1) {
        v_last = vec_t::vslideup(v_last, v_last, offset, vl);
    }
    return v_last;
}

template <typename data_vec_t, typename acc_vec_t, bool sqsum = false>
inline int integral_inner(const uchar* src_data, size_t src_step,
                          uchar* sum_data, size_t sum_step,
                          int width, int height, int cn) {
    using data_t = typename data_vec_t::ElemType;
    using acc_t = typename acc_vec_t::ElemType;

    for (int y = 0; y < height; y++) {
        const data_t* src = reinterpret_cast<const data_t*>(src_data + src_step * y);
        acc_t* prev = reinterpret_cast<acc_t*>(sum_data + sum_step * y);
        acc_t* curr = reinterpret_cast<acc_t*>(sum_data + sum_step * (y + 1));
        memset(curr, 0, cn * sizeof(acc_t));

        size_t vl = acc_vec_t::setvlmax();
        auto sum = acc_vec_t::vmv(0, vl);
        for (size_t x = 0; x < static_cast<size_t>(width); x += vl) {
            vl = acc_vec_t::setvl(width - x);
            __builtin_prefetch(&src[x + vl], 0);
            __builtin_prefetch(&prev[x + cn], 0);

            auto v_src = data_vec_t::vload(&src[x], vl);
            auto acc = acc_vec_t::cast(v_src, vl);

            if (sqsum) { // Squared Sum
                acc = acc_vec_t::vmul(acc, acc, vl);
            }

            auto v_zero = acc_vec_t::vmv(0, vl);
            for (size_t offset = cn; offset < vl; offset <<= 1) {
                auto v_shift = acc_vec_t::vslideup(v_zero, acc, offset, vl);
                acc = acc_vec_t::vadd(acc, v_shift, vl);
            }
            auto last_n = repeat_last_n<acc_vec_t>(acc, cn, vl);

            auto v_prev = acc_vec_t::vload(&prev[x + cn], vl);
            acc = acc_vec_t::vadd(acc, v_prev, vl);
            acc = acc_vec_t::vadd(acc, sum, vl);
            sum = acc_vec_t::vadd(sum, last_n, vl);

            acc_vec_t::vstore(&curr[x + cn], acc, vl);
        }
    }

    return CV_HAL_ERROR_OK;
}

template <typename data_vec_t, typename acc_vec_t, typename sq_acc_vec_t>
inline int integral(const uchar* src_data, size_t src_step, uchar* sum_data, size_t sum_step, uchar* sqsum_data, size_t sqsum_step, int width, int height, int cn) {
    memset(sum_data, 0, (sum_step) * sizeof(uchar));

    int result = CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (sqsum_data == nullptr) {
        result = integral_inner<data_vec_t, acc_vec_t, false>(src_data, src_step, sum_data, sum_step, width, height, cn);
    } else {
        result = integral_inner<data_vec_t, acc_vec_t, false>(src_data, src_step, sum_data, sum_step, width, height, cn);
        memset(sqsum_data, 0, (sqsum_step) * sizeof(uchar));
        if (result != CV_HAL_ERROR_OK) return result;
        result = integral_inner<data_vec_t, sq_acc_vec_t, true>(src_data, src_step, sqsum_data, sqsum_step, width, height, cn);
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
        result = integral<RVV<uint8_t, LMUL_1>, RVV<int32_t, LMUL_4>, RVV<double, LMUL_8>>(src_data, src_step, sum_data, sum_step, sqsum_data, sqsum_step, width, height, cn);
    else if( depth == CV_8U && sdepth == CV_32S && sqdepth == CV_32F )
        result = integral<RVV<uint8_t, LMUL_1>, RVV<int32_t, LMUL_4>, RVV<float, LMUL_4>>(src_data, src_step, sum_data, sum_step, sqsum_data, sqsum_step, width, height, cn);
    else if( depth == CV_8U && sdepth == CV_32S && sqdepth == CV_32S )
        result = integral<RVV<uint8_t, LMUL_1>, RVV<int32_t, LMUL_4>, RVV<int32_t, LMUL_4>>(src_data, src_step, sum_data, sum_step, sqsum_data, sqsum_step, width, height, cn);
    else if( depth == CV_8U && sdepth == CV_32F && sqdepth == CV_64F )
        result = integral<RVV<uint8_t, LMUL_1>, RVV<float, LMUL_4>, RVV<double, LMUL_8>>(src_data, src_step, sum_data, sum_step, sqsum_data, sqsum_step, width, height, cn);
    else if( depth == CV_8U && sdepth == CV_32F && sqdepth == CV_32F )
        result = integral<RVV<uint8_t, LMUL_1>, RVV<float, LMUL_4>, RVV<float, LMUL_4>>(src_data, src_step, sum_data, sum_step, sqsum_data, sqsum_step, width, height, cn);
    else if( depth == CV_8U && sdepth == CV_64F && sqdepth == CV_64F )
        result = integral<RVV<uint8_t, LMUL_1>, RVV<double, LMUL_8>, RVV<double, LMUL_8>>(src_data, src_step, sum_data, sum_step, sqsum_data, sqsum_step, width, height, cn);
    else if( depth == CV_16U && sdepth == CV_64F && sqdepth == CV_64F )
        result = integral<RVV<uint16_t, LMUL_1>, RVV<double, LMUL_4>, RVV<double, LMUL_4>>(src_data, src_step, sum_data, sum_step, sqsum_data, sqsum_step, width, height, cn);
    else if( depth == CV_16S && sdepth == CV_64F && sqdepth == CV_64F )
        result = integral<RVV<int16_t, LMUL_1>, RVV<double, LMUL_4>, RVV<double, LMUL_4>>(src_data, src_step, sum_data, sum_step, sqsum_data, sqsum_step, width, height, cn);
    else if( depth == CV_32F && sdepth == CV_32F && sqdepth == CV_64F )
        result = integral<RVV<float, LMUL_2>, RVV<float, LMUL_2>, RVV<double, LMUL_4>>(src_data, src_step, sum_data, sum_step, sqsum_data, sqsum_step, width, height, cn);
    else if( depth == CV_32F && sdepth == CV_32F && sqdepth == CV_32F )
        result = integral<RVV<float, LMUL_4>, RVV<float, LMUL_4>, RVV<float, LMUL_4>>(src_data, src_step, sum_data, sum_step, sqsum_data, sqsum_step, width, height, cn);
    else if( depth == CV_32F && sdepth == CV_64F && sqdepth == CV_64F )
        result = integral<RVV<float, LMUL_2>, RVV<double, LMUL_4>, RVV<double, LMUL_4>>(src_data, src_step, sum_data, sum_step, sqsum_data, sqsum_step, width, height, cn);
    else if( depth == CV_64F && sdepth == CV_64F && sqdepth == CV_64F ) {
        result = integral<RVV<double, LMUL_4>, RVV<double, LMUL_4>, RVV<double, LMUL_4>>(src_data, src_step, sum_data, sum_step, sqsum_data, sqsum_step, width, height, cn);
    }

    return result;
}

}}

#endif
