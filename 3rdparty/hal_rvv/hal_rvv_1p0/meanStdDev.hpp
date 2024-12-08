// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_MEANSTDDEV_HPP_INCLUDED
#define OPENCV_HAL_RVV_MEANSTDDEV_HPP_INCLUDED
#include <riscv_vector.h>
#include <functional>
#include <vector>

namespace cv { namespace cv_hal_rvv {
#undef cv_hal_meanStdDev
#define cv_hal_meanStdDev cv::cv_hal_rvv::meanStdDev

inline int meanStdDev_8u1c(const uchar* src_data, size_t src_step, int width, int height,
                            double* mean_val, double* stddev_val, uchar* mask, size_t mask_step);
inline int meanStdDev_32f1c(const uchar* src_data, size_t src_step, int width, int height,
                            double* mean_val, double* stddev_val, uchar* mask, size_t mask_step);
inline int meanStdDev_8u4c(const uchar* src_data, size_t src_step, int width, int height,
                            double* mean_val, double* stddev_val, uchar* mask, size_t mask_step);

inline int meanStdDev(const uchar* src_data, size_t src_step, int width, int height,
                             int src_type, double* mean_val, double* stddev_val, uchar* mask, size_t mask_step) {
    if (src_type == CV_8UC1) 
        return meanStdDev_8u1c(src_data, src_step, width, height, mean_val, stddev_val, mask, mask_step);
    else if (src_type == CV_8UC4)
        return meanStdDev_8u4c(src_data, src_step, width, height, mean_val, stddev_val, mask, mask_step);
    else if (src_type == CV_32FC1)
        return meanStdDev_32f1c(src_data, src_step, width, height, mean_val, stddev_val, mask, mask_step);
    else return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

// -----------------  meanStdDev_8u4c vector version -----------------
// Brief: vertorize 8u4c type
inline int meanStdDev_8u4c(const uchar* src_data, size_t src_step, int width, int height,
                            double* mean_val, double* stddev_val, uchar* mask, size_t mask_step) {
    double mean = 0.0, stddev = 0.0;
    for(size_t i=0; i<4; ++i) {
        const uchar* src_c = src_data + i;
        double* cur_mean, *cur_stddev;
        if(!stddev_val) cur_stddev = nullptr;
        meanStdDev_8u1c(src_c, src_step, width, height, cur_mean, cur_stddev, mask, mask_step);
        mean += *cur_mean;
        if(!stddev_val) stddev += *cur_stddev;
    }
    *mean_val = mean;
    *stddev_val = stddev;
    return CV_HAL_ERROR_OK;
}

// -----------------  meanStdDev_8u1c vector final version  -----------------
// Brief: loop unroll for outter loop
inline int meanStdDev_8u1c(const uchar* src_data, size_t src_step, int width, int height,
                            double* mean_val, double* stddev_val, uchar* mask, size_t mask_step) {
    size_t loop_unroll = 4;
    size_t loop_rem = height % loop_unroll; 
    // initialize variables
    size_t total_count = 0;
    size_t vlmax = __riscv_vsetvlmax_e8m1();
    vuint64m1_t vec_sum = __riscv_vmv_v_x_u64m1(0, vlmax);
    vuint64m1_t vec_sqsum = __riscv_vmv_v_x_u64m1(0, vlmax);
    vuint16m1_t vec_s = __riscv_vmv_v_x_u16m1(0, vlmax);
    vuint32m1_t temp_sqsum = __riscv_vmv_v_x_u32m1(0, vlmax);
    int j = 0;
    int vl = __riscv_vsetvl_e8m1(width-j);
    size_t i=0;
    for ( ; i < height - loop_rem; i+=4) {
        const uchar* src_row1 = src_data + i * src_step;
        const uchar* src_row2 = src_data + ( i + 1 ) * src_step;
        const uchar* src_row3 = src_data + ( i + 2 ) * src_step;
        const uchar* src_row4 = src_data + ( i + 3 ) * src_step;

        const uchar* mask_row1 = mask ? (mask + i * mask_step) : nullptr;
        const uchar* mask_row2 = mask ? (mask + ( i + 1 ) * mask_step) : nullptr;
        const uchar* mask_row3 = mask ? (mask + ( i + 2 ) * mask_step) : nullptr;
        const uchar* mask_row4 = mask ? (mask + ( i + 3 ) * mask_step) : nullptr;

        for ( ; j < width; ) {
            vl = __riscv_vsetvl_e8m1(width-j); // tail elements

            vuint16m2_t pixel_squared_vector1;
            vuint16m2_t pixel_squared_vector2;
            vuint16m2_t pixel_squared_vector3;
            vuint16m2_t pixel_squared_vector4;

            // Load src[row][i .. i+vl]
            vuint8m1_t pixel_vector1 = __riscv_vle8_v_u8m1(src_row1 + j, vl);
            vuint8m1_t pixel_vector2 = __riscv_vle8_v_u8m1(src_row2 + j, vl);
            vuint8m1_t pixel_vector3 = __riscv_vle8_v_u8m1(src_row3 + j, vl);
            vuint8m1_t pixel_vector4 = __riscv_vle8_v_u8m1(src_row4 + j, vl);

            if(mask) {
                // Load mask[row][i .. i+vl]
                vbool8_t mask_vector1 = __riscv_vlm_v_b8(mask_row1 + j, vl);
                vbool8_t mask_vector2 = __riscv_vlm_v_b8(mask_row2 + j, vl);
                vbool8_t mask_vector3 = __riscv_vlm_v_b8(mask_row3 + j, vl);
                vbool8_t mask_vector4 = __riscv_vlm_v_b8(mask_row4 + j, vl);

                // vec_s[0] <- sum(vec_s[0] , pixel_vector[*]) , if not masked
                vec_s = __riscv_vwredsumu_vs_u8m1_u16m1_m(mask_vector1, pixel_vector1, vec_s, vl);
                vec_s = __riscv_vwredsumu_vs_u8m1_u16m1_m(mask_vector2, pixel_vector2, vec_s, vl);
                vec_s = __riscv_vwredsumu_vs_u8m1_u16m1_m(mask_vector3, pixel_vector3, vec_s, vl);
                vec_s = __riscv_vwredsumu_vs_u8m1_u16m1_m(mask_vector4, pixel_vector4, vec_s, vl);

                if(stddev_val) {
                    // pixel_squared_vector[*] = pixel_vector[*] * pixel_vector[*] , if not masked
                    pixel_squared_vector1 = __riscv_vwmulu_vv_u16m2_m(mask_vector1, pixel_vector1, pixel_vector1, vl);
                    pixel_squared_vector2 = __riscv_vwmulu_vv_u16m2_m(mask_vector2, pixel_vector2, pixel_vector2, vl);
                    pixel_squared_vector3 = __riscv_vwmulu_vv_u16m2_m(mask_vector3, pixel_vector3, pixel_vector3, vl);
                    pixel_squared_vector4 = __riscv_vwmulu_vv_u16m2_m(mask_vector4, pixel_vector4, pixel_vector4, vl);
                    // sqsum[0] <- sum(temp_sqsum[0] , pixel_squared_vector[*]) , if not masked
                    temp_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1_m(mask_vector1, pixel_squared_vector1, temp_sqsum, vl);
                    temp_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1_m(mask_vector2, pixel_squared_vector2, temp_sqsum, vl);
                    temp_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1_m(mask_vector3, pixel_squared_vector3, temp_sqsum, vl);
                    temp_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1_m(mask_vector4, pixel_squared_vector4, temp_sqsum, vl);
                }
            }
            else {
                // vec_s[0] <- sum(vec_u16m1_zero[0] , pixel_vector[*])
                vec_s = __riscv_vwredsumu_vs_u8m1_u16m1(pixel_vector1, vec_s, vl);
                vec_s = __riscv_vwredsumu_vs_u8m1_u16m1(pixel_vector2, vec_s, vl);
                vec_s = __riscv_vwredsumu_vs_u8m1_u16m1(pixel_vector3, vec_s, vl);
                vec_s = __riscv_vwredsumu_vs_u8m1_u16m1(pixel_vector4, vec_s, vl);
                if(stddev_val) {
                    // pixel_squared_vector[*] = pixel_vector[*] * pixel_vector[*]
                    pixel_squared_vector1 = __riscv_vwmulu_vv_u16m2(pixel_vector1, pixel_vector1, vl);
                    pixel_squared_vector2 = __riscv_vwmulu_vv_u16m2(pixel_vector2, pixel_vector2, vl);
                    pixel_squared_vector3 = __riscv_vwmulu_vv_u16m2(pixel_vector3, pixel_vector3, vl);
                    pixel_squared_vector4 = __riscv_vwmulu_vv_u16m2(pixel_vector4, pixel_vector4, vl);
                    // sqsum[0] <- sum(vec_u32m2_zero[0] , pixel_squared_vector[*])
                    temp_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1(pixel_squared_vector1, temp_sqsum, vl);
                    temp_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1(pixel_squared_vector2, temp_sqsum, vl);
                    temp_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1(pixel_squared_vector3, temp_sqsum, vl);
                    temp_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1(pixel_squared_vector4, temp_sqsum, vl);
                }
            }

            // vuint64m1_t <- vuint16m1_t
            vuint64m1_t temp_sum = __riscv_vreinterpret_u64m1(vec_s);
            // vec_sum[0] = sum( temp_sum , vec_sum)
            vec_sum = __riscv_vadd_vv_u64m1(temp_sum, vec_sum, vl);
            if (stddev_val) {
                temp_sum = __riscv_vreinterpret_u64m1(temp_sqsum);
                vec_sqsum = __riscv_vadd_vv_u64m1(temp_sum, vec_sqsum, vl);
            }
            total_count += vl;
            j += vl;
        }
    }
    for ( ; i < height; ++i) {
        const uchar* src_row = src_data + i * src_step;
        const uchar* mask_row = mask ? (mask + i * mask_step) : nullptr;

        for ( ; j < width; ) {
            vl = __riscv_vsetvl_e8m1(width-j); // tail elements

            // Load src[row][i .. i+vl]
            vuint8m1_t pixel_vector = __riscv_vle8_v_u8m1(src_row + j, vl);
            vuint16m2_t pixel_squared_vector;
            if(mask) {
                // Load mask[row][i .. i+vl]
                vbool8_t mask_vector = __riscv_vlm_v_b8(mask_row + j, vl);

                // vec_s[0] <- sum(vec_u16m1_zero[0] , pixel_vector[*]) , if not masked
                vec_s = __riscv_vwredsumu_vs_u8m1_u16m1_m(mask_vector, pixel_vector, vec_s, vl);
                if(stddev_val) {
                    // pixel_squared_vector[*] = pixel_vector[*] * pixel_vector[*] , if not masked
                    pixel_squared_vector = __riscv_vwmulu_vv_u16m2_m(mask_vector, pixel_vector, pixel_vector, vl);
                    // sqsum[0] <- sum(vec_u32m2_zero[0] , pixel_squared_vector[*]) , if not masked
                    temp_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1_m(mask_vector, pixel_squared_vector, temp_sqsum, vl);
                }
            }
            else {
                // vec_s[0] <- sum(vec_u16m1_zero[0] , pixel_vector[*])
                vec_s = __riscv_vwredsumu_vs_u8m1_u16m1(pixel_vector, vec_s, vl);
                if(stddev_val) {
                    // pixel_squared_vector[*] = pixel_vector[*] * pixel_vector[*]
                    pixel_squared_vector = __riscv_vwmulu_vv_u16m2(pixel_vector, pixel_vector, vl);
                    // sqsum[0] <- sum(vec_u32m2_zero[0] , pixel_squared_vector[*])
                    temp_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1(pixel_squared_vector, temp_sqsum, vl);
                }
            }
            // vuint64m1_t <- vuint16m1_t
            vuint64m1_t temp_sum = __riscv_vreinterpret_u64m1(vec_s);
            // vec_sum[0] = sum( temp_sum , vec_sum)
            vec_sum = __riscv_vadd_vv_u64m1(temp_sum, vec_sum, vl);
            if (stddev_val) {
                temp_sum = __riscv_vreinterpret_u64m1(temp_sqsum);
                vec_sqsum = __riscv_vadd_vv_u64m1(temp_sum, vec_sqsum, vl);
            }
            total_count += vl;
            j += vl;
        }
    }

    if (total_count == 0)
    {
        if (mean_val) *mean_val = 0.0;
        if (stddev_val) *stddev_val = 0.0;
        return CV_HAL_ERROR_OK;
    }
    // Return values
    vfloat64m1_t float_sum = __riscv_vreinterpret_f64m1(vec_sum);
    double sum = __riscv_vfmv_f_s_f64m1_f64(float_sum);
    double mean = sum / total_count;
    if (mean_val) {
        *mean_val = mean;
    }
    if (stddev_val) {
        vfloat64m1_t float_sqsum = __riscv_vreinterpret_f64m1(vec_sqsum);
        double sqsum = __riscv_vfmv_f_s_f64m1_f64(float_sqsum);
        double variance = std::max((sqsum / total_count) - (mean * mean), 0.0);
        double stddev = std::sqrt(variance);
        *stddev_val = stddev;
    }
    return CV_HAL_ERROR_OK;
}

// -----------------  meanStdDev_f321c vector version  -----------------
// Brief: vertorize 32f1c type
inline int meanStdDev_32f1c(const uchar* src_data, size_t src_step, int width, int height,
                            double* mean_val, double* stddev_val, uchar* mask, size_t mask_step) {
    size_t loop_unroll = 4;
    size_t loop_rem = height % loop_unroll;
    // initialize variables
    size_t total_count = 0;
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t vec_sum = __riscv_vfmv_v_f_f32m1(0.0, vlmax);
    vfloat32m1_t vec_sqsum = __riscv_vfmv_v_f_f32m1(0.0, vlmax);
    vfloat32m1_t vec_s = __riscv_vfmv_v_f_f32m1(0.0, vlmax);
    vfloat32m1_t temp_sqsum = __riscv_vfmv_v_f_f32m1(0.0, vlmax);
    vfloat64m1_t vec_f64_zero = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    int j = 0;
    int vl = __riscv_vsetvl_e8m1(width-j);
    size_t i=0;
    for ( ; i < height - loop_rem; i+=4) {
        const float* src_row1 = reinterpret_cast<const float*>(src_data) + i * src_step;
        const float* src_row2 = reinterpret_cast<const float*>(src_data) + (i+1) * src_step;
        const float* src_row3 = reinterpret_cast<const float*>(src_data) + (i+2) * src_step;
        const float* src_row4 = reinterpret_cast<const float*>(src_data) + (i+3) * src_step;
        const uchar* mask_row1 = mask ? (mask + i * mask_step) : nullptr;
        const uchar* mask_row2 = mask ? (mask + ( i + 1 ) * mask_step) : nullptr;
        const uchar* mask_row3 = mask ? (mask + ( i + 2 ) * mask_step) : nullptr;
        const uchar* mask_row4 = mask ? (mask + ( i + 3 ) * mask_step) : nullptr;
        for ( ; j < width; ) {
            vl = __riscv_vsetvl_e8m1(width-j); // tail elements
            // Load src[row][i .. i+vl]
            vfloat32m1_t pixel_vector1 = __riscv_vle32_v_f32m4_f32m_4_m(src_row1 + j, vl);
            vfloat32m1_t pixel_vector2 = __riscv_vle32_v_f32m1(src_row2 + j, vl);
            vfloat32m1_t pixel_vector3 = __riscv_vle32_v_f32m1(src_row3 + j, vl);
            vfloat32m1_t pixel_vector4 = __riscv_vle32_v_f32m1(src_row4 + j, vl);
            vfloat32m1_t pixel_squared_vector1;
            vfloat32m1_t pixel_squared_vector2;
            vfloat32m1_t pixel_squared_vector3;
            vfloat32m1_t pixel_squared_vector4;
            if(mask) {
                // Load mask[row][i .. i+vl]
                vbool32_t mask_vector1 = __riscv_vlm_v_b32(mask_row1 + j, vl);
                vbool32_t mask_vector2 = __riscv_vlm_v_b32(mask_row2 + j, vl);
                vbool32_t mask_vector3 = __riscv_vlm_v_b32(mask_row3 + j, vl);
                vbool32_t mask_vector4 = __riscv_vlm_v_b32(mask_row4 + j, vl);
                // vec_s[0] <- sum(vec_s[0] , pixel_vector[*]) , if not masked
                vec_s = __riscv_vfredusum_vs_f32m1_f32m1_m(mask_vector1, pixel_vector1, vec_s, vl);
                vec_s = __riscv_vfredusum_vs_f32m1_f32m1_m(mask_vector2, pixel_vector2, vec_s, vl);
                vec_s = __riscv_vfredusum_vs_f32m1_f32m1_m(mask_vector3, pixel_vector3, vec_s, vl);
                vec_s = __riscv_vfredusum_vs_f32m1_f32m1_m(mask_vector4, pixel_vector4, vec_s, vl);
                if(stddev_val) {
                    // pixel_squared_vector[*] = pixel_vector[*] * pixel_vector[*] , if not masked
                    pixel_squared_vector1 = __riscv_vfmul_vv_f32m1_m(mask_vector1, pixel_vector1, pixel_vector1, vl);
                    pixel_squared_vector2 = __riscv_vfmul_vv_f32m1_m(mask_vector2, pixel_vector2, pixel_vector2, vl);
                    pixel_squared_vector3 = __riscv_vfmul_vv_f32m1_m(mask_vector3, pixel_vector3, pixel_vector3, vl);
                    pixel_squared_vector4 = __riscv_vfmul_vv_f32m1_m(mask_vector4, pixel_vector4, pixel_vector4, vl);
                    // sqsum[0] <- sum(temp_sqsum[0] , pixel_squared_vector[*]) , if not masked
                    temp_sqsum = __riscv_vfredusum_vs_f32m1_f32m1_m(mask_vector1, pixel_squared_vector1, temp_sqsum, vl);
                    temp_sqsum = __riscv_vfredusum_vs_f32m1_f32m1_m(mask_vector2, pixel_squared_vector2, temp_sqsum, vl);
                    temp_sqsum = __riscv_vfredusum_vs_f32m1_f32m1_m(mask_vector3, pixel_squared_vector3, temp_sqsum, vl);
                    temp_sqsum = __riscv_vfredusum_vs_f32m1_f32m1_m(mask_vector4, pixel_squared_vector4, temp_sqsum, vl);
                }
            }
            else {
                // vec_s[0] <- sum(vec_u16m1_zero[0] , pixel_vector[*])
                vec_s = __riscv_vfredusum_vs_f32m1_f32m1(pixel_vector1, vec_s, vl);
                vec_s = __riscv_vfredusum_vs_f32m1_f32m1(pixel_vector2, vec_s, vl);
                vec_s = __riscv_vfredusum_vs_f32m1_f32m1(pixel_vector3, vec_s, vl);
                vec_s = __riscv_vfredusum_vs_f32m1_f32m1(pixel_vector4, vec_s, vl);
                if(stddev_val) {
                    // pixel_squared_vector[*] = pixel_vector[*] * pixel_vector[*]
                    pixel_squared_vector1 = __riscv_vfmul_vv_f32m1(pixel_vector1, pixel_vector1, vl);
                    pixel_squared_vector2 = __riscv_vfmul_vv_f32m1(pixel_vector2, pixel_vector2, vl);
                    pixel_squared_vector3 = __riscv_vfmul_vv_f32m1(pixel_vector3, pixel_vector3, vl);
                    pixel_squared_vector4 = __riscv_vfmul_vv_f32m1(pixel_vector4, pixel_vector4, vl);
                    // sqsum[0] <- sum(vec_u32m2_zero[0] , pixel_squared_vector[*])
                    temp_sqsum = __riscv_vfredusum_vs_f32m1_f32m1(pixel_squared_vector1, temp_sqsum, vl);
                    temp_sqsum = __riscv_vfredusum_vs_f32m1_f32m1(pixel_squared_vector2, temp_sqsum, vl);
                    temp_sqsum = __riscv_vfredusum_vs_f32m1_f32m1(pixel_squared_vector3, temp_sqsum, vl);
                    temp_sqsum = __riscv_vfredusum_vs_f32m1_f32m1(pixel_squared_vector4, temp_sqsum, vl);
                }
            }
            // vec_sum[0] = sum( temp_sum , vec_sum)
            vec_sum = __riscv_vfadd_vv_f32m1(vec_s, vec_sum, vl);
            if (stddev_val) {
                vec_sqsum = __riscv_vfadd_vv_f32m1(temp_sqsum, vec_sqsum, vl);
            }
            total_count += vl;
            j += vl;
        }
    }
    for ( ; i < height; ++i) {
        const float* src_row = reinterpret_cast<const float*>(src_data) + i * src_step;
        const uchar* mask_row = mask ? (mask + i * mask_step) : nullptr;
        for ( ; j < width; ) {
            vl = __riscv_vsetvl_e32m1(width-j); // tail elements
            // Load src[row][i .. i+vl]
            vfloat32m1_t pixel_vector = __riscv_vle32_v_f32m1(src_row + j, vl);
            vfloat32m1_t pixel_squared_vector;
            if(mask) {
                // Load mask[row][i .. i+vl]
                vbool32_t mask_vector = __riscv_vlm_v_b32(mask_row + j, vl);
                // vec_s[0] <- sum(vec_u16m1_zero[0] , pixel_vector[*]) , if not masked
                vec_s = __riscv_vfredusum_vs_f32m1_f32m1_m(mask_vector, pixel_vector, vec_s, vl);
                if(stddev_val) {
                    // pixel_squared_vector[*] = pixel_vector[*] * pixel_vector[*] , if not masked
                    pixel_squared_vector = __riscv_vfmul_vv_f32m1_m(mask_vector, pixel_vector, pixel_vector, vl);
                    // sqsum[0] <- sum(vec_u32m2_zero[0] , pixel_squared_vector[*]) , if not masked
                    temp_sqsum = __riscv_vfredusum_vs_f32m1_f32m1_m(mask_vector, pixel_squared_vector, temp_sqsum, vl);
                }
            }
            else {
                // vec_s[0] <- sum(vec_u16m1_zero[0] , pixel_vector[*])
                vec_s = __riscv_vfredusum_vs_f32m1_f32m1(pixel_vector, vec_s, vl);
                if(stddev_val) {
                    // pixel_squared_vector[*] = pixel_vector[*] * pixel_vector[*]
                    pixel_squared_vector = __riscv_vfmul_vv_f32m1(pixel_vector, pixel_vector, vl);
                    // sqsum[0] <- sum(vec_u32m2_zero[0] , pixel_squared_vector[*])
                    temp_sqsum = __riscv_vfredusum_vs_f32m1_f32m1(pixel_squared_vector, temp_sqsum, vl);
                }
            }
            // vec_sum[0] = sum( temp_sum , vec_sum)
            vec_sum = __riscv_vfadd_vv_f32m1(vec_s, vec_sum, vl);
            if (stddev_val) {
                vec_sqsum = __riscv_vfadd_vv_f32m1(temp_sqsum, vec_sqsum, vl);
            }
            total_count += vl;
            j += vl;
        }
    }
    if (total_count == 0)
    {
        if (mean_val) *mean_val = 0.0;
        if (stddev_val) *stddev_val = 0.0;
        return CV_HAL_ERROR_OK;
    }
    // Return values
    vfloat64m1_t float_sum = __riscv_vfwredusum_vs_f32m1_f64m1(vec_sum, vec_f64_zero, vl);
    double sum = __riscv_vfmv_f_s_f64m1_f64(float_sum);
    double mean = sum / total_count;
    if (mean_val) {
        *mean_val = mean;
    }
    if (stddev_val) {
        vfloat64m1_t float_sqsum = __riscv_vfwredusum_vs_f32m1_f64m1(vec_sum, vec_f64_zero, vl);
        double sqsum = __riscv_vfmv_f_s_f64m1_f64(float_sqsum);
        double variance = std::max((sqsum / total_count) - (mean * mean), 0.0);
        double stddev = std::sqrt(variance);
        *stddev_val = stddev;
    }
    return CV_HAL_ERROR_OK;
}

// -----------------  meanStdDev_8u1c scalar version  -----------------
// inline int meanStdDev_8u1c(const uchar* src_data, size_t src_step, int width, int height,
//                             double* mean_val, double* stddev_val, uchar* mask, size_t mask_step) {
//     double sum = 0.0, sqsum = 0.0;
//     size_t total_count = 0;
//     for (int i = 0; i < height; ++i)
//     {
//         const uchar* src_row = src_data + i * src_step;
//         const uchar* mask_row = mask ? (mask + i * mask_step) : nullptr;
//         for (int j = 0; j < width; ++j)
//         {
//             if (mask_row && mask_row[j] == 0)
//                 continue;
//             double pixel_value = static_cast<double>(src_row[j]);
//             sum += pixel_value;
//             if (stddev_val)
//                 sqsum += pixel_value * pixel_value;
//             ++total_count;
//         }
//     }
//     if (total_count == 0)
//     {
//         if (mean_val) *mean_val = 0.0;
//         if (stddev_val) *stddev_val = 0.0;
//         return CV_HAL_ERROR_OK;
//     }
//     double mean = sum / total_count;
//     if (mean_val) {
//         *mean_val = mean;
//     }
//     if (stddev_val) {
//         double variance = std::max((sqsum / total_count) - (mean * mean), 0.0);
//         double stddev = std::sqrt(variance);
//         *stddev_val = stddev;
//     }
//     return CV_HAL_ERROR_OK;
// }

// -----------------  meanStdDev_8u1c vector version 1.0  -----------------
// Brief: vectorize inner loop
// inline int meanStdDev_8u1c(const uchar* src_data, size_t src_step, int width, int height,
//                              double* mean_val, double* stddev_val, uchar* mask, size_t mask_step) {
//     // initialize variables
//     size_t total_count = 0;
//     size_t vlmax = __riscv_vsetvlmax_e8m1();
//     vuint16m1_t vec_u16m1_zero = __riscv_vmv_v_x_u16m1(0, vlmax);
//     vuint32m1_t vec_u32m1_zero = __riscv_vmv_v_x_u32m1(0, vlmax);
//     vuint64m1_t vec_sum = __riscv_vmv_v_x_u64m1(0, vlmax);
//     vuint64m1_t vec_sqsum = __riscv_vmv_v_x_u64m1(0, vlmax);
//     for (int i = 0; i < height; ++i)
//     {
//         const uchar* src_row = src_data + i * src_step;
//         const uchar* mask_row = mask ? (mask + i * mask_step) : nullptr;
//         int j = 0;
//         int vl = __riscv_vsetvl_e8m1(width-j);
//         for ( ; j < width; ) {
//             if (mask_row && mask_row[j] == 0) {
//                 ++j;
//                 continue;
//             }
//             vl = __riscv_vsetvl_e8m1(width-j); // tail elements
//             // Load src[row][i .. i+vl]
//             vuint8m1_t pixel_vector = __riscv_vle8_v_u8m1(src_row + j, vl);
//             // vec_s[0] <- sum(vec_u16m1_zero[0] , pixel_vector[*]) 
//             vuint16m1_t vec_s = __riscv_vwredsumu_vs_u8m1_u16m1(pixel_vector, vec_u16m1_zero, vl);
//             // vuint64m1_t <- vuint16m1_t
//             vuint64m1_t temp_sum = __riscv_vreinterpret_u64m1(vec_s);
//             // vec_sum[0] = sum( temp_sum , vec_sum)
//             vec_sum = __riscv_vadd_vv_u64m1(temp_sum, vec_sum, vl);
//             if (stddev_val) {
//                 // pixel_squared_vector[*] = pixel_vector[*] * pixel_vector[*]
//                 vuint16m2_t pixel_squared_vector = __riscv_vwmulu_vv_u16m2(pixel_vector, pixel_vector, vl);
//                 // sqsum[0] <- sum(vec_u32m2_zero[0] , pixel_squared_vector[*])
//                 vuint32m1_t temp_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1(pixel_squared_vector, vec_u32m1_zero, vl);
//                 temp_sum = __riscv_vreinterpret_u64m1(temp_sqsum);
//                 vec_sqsum = __riscv_vadd_vv_u64m1(temp_sum, vec_sqsum, vl);
//             }
//             total_count += vl;
//             j += vl;
//         }
//     }
//     if (total_count == 0)
//     {
//         if (mean_val) *mean_val = 0.0;
//         if (stddev_val) *stddev_val = 0.0;
//         return CV_HAL_ERROR_OK;
//     }
//     // Return values
//     vfloat64m1_t float_sum = __riscv_vreinterpret_f64m1(vec_sum);
//     double sum = __riscv_vfmv_f_s_f64m1_f64(float_sum);
//     double mean = sum / total_count;
//     if (mean_val) {
//         *mean_val = mean;
//     }
//     if (stddev_val) {
//         vfloat64m1_t float_sqsum = __riscv_vreinterpret_f64m1(vec_sqsum);
//         double sqsum = __riscv_vfmv_f_s_f64m1_f64(float_sqsum);
//         double variance = std::max((sqsum / total_count) - (mean * mean), 0.0);
//         double stddev = std::sqrt(variance);
//         *stddev_val = stddev;
//     }
//     return CV_HAL_ERROR_OK;
// }

// -----------------  meanStdDev_8u1c vector version 1.1  -----------------
// Brief: vectorize mask
// inline int meanStdDev_8u1c(const uchar* src_data, size_t src_step, int width, int height,
//                              double* mean_val, double* stddev_val, uchar* mask, size_t mask_step) {
//     // initialize variables
//     size_t total_count = 0;
//     size_t vlmax = __riscv_vsetvlmax_e8m1();
//     vuint16m1_t vec_u16m1_zero = __riscv_vmv_v_x_u16m1(0, vlmax);
//     vuint32m1_t vec_u32m1_zero = __riscv_vmv_v_x_u32m1(0, vlmax);
//     vuint64m1_t vec_sum = __riscv_vmv_v_x_u64m1(0, vlmax);
//     vuint64m1_t vec_sqsum = __riscv_vmv_v_x_u64m1(0, vlmax);
//     for (int i = 0; i < height; ++i)
//     {
//         const uchar* src_row = src_data + i * src_step;
//         const uchar* mask_row = mask ? (mask + i * mask_step) : nullptr;
//         int j = 0;
//         int vl = __riscv_vsetvl_e8m1(width-j);
//         for ( ; j < width; ) {
//             vl = __riscv_vsetvl_e8m1(width-j); // tail elements
//
//             // Load src[row][i .. i+vl]
//             vuint8m1_t pixel_vector = __riscv_vle8_v_u8m1(src_row + j, vl);
//             // Load mask[row][i .. i+vl]
//             vuint16m1_t vec_s;
//             vuint16m2_t pixel_squared_vector;
//             vuint32m1_t temp_sqsum;
//             if(mask_row) {
//                 vbool8_t mask_vector = __riscv_vlm_v_b8(mask_row + j, vl);
//
//                 // vec_s[0] <- sum(vec_u16m1_zero[0] , pixel_vector[*]) , if not masked
//                 vec_s = __riscv_vwredsumu_vs_u8m1_u16m1_m(mask_vector, pixel_vector, vec_u16m1_zero, vl);
//                 if(stddev_val) {
//                     // pixel_squared_vector[*] = pixel_vector[*] * pixel_vector[*] , if not masked
//                     pixel_squared_vector = __riscv_vwmulu_vv_u16m2_m(mask_vector, pixel_vector, pixel_vector, vl);
//                     // sqsum[0] <- sum(vec_u32m2_zero[0] , pixel_squared_vector[*]) , if not masked
//                     temp_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1_m(mask_vector, pixel_squared_vector, vec_u32m1_zero, vl);
//                 }
//             }
//             else {
//                 // vec_s[0] <- sum(vec_u16m1_zero[0] , pixel_vector[*])
//                 vec_s = __riscv_vwredsumu_vs_u8m1_u16m1(pixel_vector, vec_u16m1_zero, vl);
//                 if(stddev_val) {
//                     // pixel_squared_vector[*] = pixel_vector[*] * pixel_vector[*]
//                     pixel_squared_vector = __riscv_vwmulu_vv_u16m2(pixel_vector, pixel_vector, vl);
//                     // sqsum[0] <- sum(vec_u32m2_zero[0] , pixel_squared_vector[*])
//                     temp_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1(pixel_squared_vector, vec_u32m1_zero, vl);
//                 }
//             }
//             // vuint64m1_t <- vuint16m1_t
//             vuint64m1_t temp_sum = __riscv_vreinterpret_u64m1(vec_s);
//             // vec_sum[0] = sum( temp_sum , vec_sum)
//             vec_sum = __riscv_vadd_vv_u64m1(temp_sum, vec_sum, vl);
//             if (stddev_val) {
//                 temp_sum = __riscv_vreinterpret_u64m1(temp_sqsum);
//                 vec_sqsum = __riscv_vadd_vv_u64m1(temp_sum, vec_sqsum, vl);
//             }
//             total_count += vl;
//             j += vl;
//         }
//     }
//     if (total_count == 0)
//     {
//         if (mean_val) *mean_val = 0.0;
//         if (stddev_val) *stddev_val = 0.0;
//         return CV_HAL_ERROR_OK;
//     }
//     // Return values
//     vfloat64m1_t float_sum = __riscv_vreinterpret_f64m1(vec_sum);
//     double sum = __riscv_vfmv_f_s_f64m1_f64(float_sum);
//     double mean = sum / total_count;
//     if (mean_val) {
//         *mean_val = mean;
//     }
//     if (stddev_val) {
//         vfloat64m1_t float_sqsum = __riscv_vreinterpret_f64m1(vec_sqsum);
//         double sqsum = __riscv_vfmv_f_s_f64m1_f64(float_sqsum);
//         double variance = std::max((sqsum / total_count) - (mean * mean), 0.0);
//         double stddev = std::sqrt(variance);
//         *stddev_val = stddev;
//     }
//     return CV_HAL_ERROR_OK;
// }

// -----------------  meanStdDev_8u1c vector version 1.3  -----------------
// Brief: expand lmul
// inline int meanStdDev_8u1c(const uchar* src_data, size_t src_step, int width, int height,
//                             double* mean_val, double* stddev_val, uchar* mask, size_t mask_step) {
//     size_t loop_unroll = 4;
//     size_t loop_rem = height % loop_unroll; 
//     // initialize variables
//     size_t total_count = 0;
//     size_t vlmax = __riscv_vsetvlmax_e8m1();
//     vuint64m1_t vec_sum = __riscv_vmv_v_x_u64m1(0, vlmax);
//     vuint64m1_t vec_sqsum = __riscv_vmv_v_x_u64m1(0, vlmax);
//     vuint16m1_t vec_s = __riscv_vmv_v_x_u16m1(0, vlmax);
//     vuint32m1_t temp_sqsum = __riscv_vmv_v_x_u32m1(0, vlmax);
//     int j = 0;
//     int vl = __riscv_vsetvl_e8m1(width-j);
//     size_t i=0;
//     for ( ; i < height - loop_rem; i+=4) {
//         const uchar* src_row1 = src_data + i * src_step;
//         const uchar* src_row2 = src_data + ( i + 1 ) * src_step;
//         const uchar* src_row3 = src_data + ( i + 2 ) * src_step;
//         const uchar* src_row4 = src_data + ( i + 3 ) * src_step;
//         const uchar* mask_row1 = mask ? (mask + i * mask_step) : nullptr;
//         const uchar* mask_row2 = mask ? (mask + ( i + 1 ) * mask_step) : nullptr;
//         const uchar* mask_row3 = mask ? (mask + ( i + 2 ) * mask_step) : nullptr;
//         const uchar* mask_row4 = mask ? (mask + ( i + 3 ) * mask_step) : nullptr;
//         for ( ; j < width; ) {
//             vl = __riscv_vsetvl_e8m1(width-j); // tail elements
//             vuint16m4_t pixel_squared_vector1;
//             vuint16m4_t pixel_squared_vector2;
//             vuint16m4_t pixel_squared_vector3;
//             vuint16m4_t pixel_squared_vector4;
//             // Load src[row][i .. i+vl]
//             vuint8m2_t pixel_vector1 = __riscv_vle8_v_u8m2(src_row1 + j, vl);
//             vuint8m2_t pixel_vector2 = __riscv_vle8_v_u8m2(src_row2 + j, vl);
//             vuint8m2_t pixel_vector3 = __riscv_vle8_v_u8m2(src_row3 + j, vl);
//             vuint8m2_t pixel_vector4 = __riscv_vle8_v_u8m2(src_row4 + j, vl);
//             if(mask) {
//                 // Load mask[row][i .. i+vl]
//                 vbool4_t mask_vector1 = __riscv_vlm_v_b4(mask_row1 + j, vl);
//                 vbool4_t mask_vector2 = __riscv_vlm_v_b4(mask_row2 + j, vl);
//                 vbool4_t mask_vector3 = __riscv_vlm_v_b4(mask_row3 + j, vl);
//                 vbool4_t mask_vector4 = __riscv_vlm_v_b4(mask_row4 + j, vl);
//                 // vec_s[0] <- sum(vec_s[0] , pixel_vector[*]) , if not masked
//                 vec_s = __riscv_vwredsumu_vs_u8m2_u16m1_m(mask_vector1, pixel_vector1, vec_s, vl);
//                 vec_s = __riscv_vwredsumu_vs_u8m2_u16m1_m(mask_vector2, pixel_vector2, vec_s, vl);
//                 vec_s = __riscv_vwredsumu_vs_u8m2_u16m1_m(mask_vector3, pixel_vector3, vec_s, vl);
//                 vec_s = __riscv_vwredsumu_vs_u8m2_u16m1_m(mask_vector4, pixel_vector4, vec_s, vl);
//                 if(stddev_val) {
//                     // pixel_squared_vector[*] = pixel_vector[*] * pixel_vector[*] , if not masked
//                     pixel_squared_vector1 = __riscv_vwmulu_vv_u16m4_m(mask_vector1, pixel_vector1, pixel_vector1, vl);
//                     pixel_squared_vector2 = __riscv_vwmulu_vv_u16m4_m(mask_vector2, pixel_vector2, pixel_vector2, vl);
//                     pixel_squared_vector3 = __riscv_vwmulu_vv_u16m4_m(mask_vector3, pixel_vector3, pixel_vector3, vl);
//                     pixel_squared_vector4 = __riscv_vwmulu_vv_u16m4_m(mask_vector4, pixel_vector4, pixel_vector4, vl);
//                     // sqsum[0] <- sum(temp_sqsum[0] , pixel_squared_vector[*]) , if not masked
//                     temp_sqsum = __riscv_vwredsumu_vs_u16m4_u32m1_m(mask_vector1, pixel_squared_vector1, temp_sqsum, vl);
//                     temp_sqsum = __riscv_vwredsumu_vs_u16m4_u32m1_m(mask_vector2, pixel_squared_vector2, temp_sqsum, vl);
//                     temp_sqsum = __riscv_vwredsumu_vs_u16m4_u32m1_m(mask_vector3, pixel_squared_vector3, temp_sqsum, vl);
//                     temp_sqsum = __riscv_vwredsumu_vs_u16m4_u32m1_m(mask_vector4, pixel_squared_vector4, temp_sqsum, vl);
//                 }
//             }
//             else {
//                 // vec_s[0] <- sum(vec_u16m1_zero[0] , pixel_vector[*])
//                 vec_s = __riscv_vwredsumu_vs_u8m2_u16m1(pixel_vector1, vec_s, vl);
//                 vec_s = __riscv_vwredsumu_vs_u8m2_u16m1(pixel_vector2, vec_s, vl);
//                 vec_s = __riscv_vwredsumu_vs_u8m2_u16m1(pixel_vector3, vec_s, vl);
//                 vec_s = __riscv_vwredsumu_vs_u8m2_u16m1(pixel_vector4, vec_s, vl);
//                 if(stddev_val) {
//                     // pixel_squared_vector[*] = pixel_vector[*] * pixel_vector[*]
//                     pixel_squared_vector1 = __riscv_vwmulu_vv_u16m4(pixel_vector1, pixel_vector1, vl);
//                     pixel_squared_vector2 = __riscv_vwmulu_vv_u16m4(pixel_vector2, pixel_vector2, vl);
//                     pixel_squared_vector3 = __riscv_vwmulu_vv_u16m4(pixel_vector3, pixel_vector3, vl);
//                     pixel_squared_vector4 = __riscv_vwmulu_vv_u16m4(pixel_vector4, pixel_vector4, vl);
//                     // sqsum[0] <- sum(vec_u32m2_zero[0] , pixel_squared_vector[*])
//                     temp_sqsum = __riscv_vwredsumu_vs_u16m4_u32m1(pixel_squared_vector1, temp_sqsum, vl);
//                     temp_sqsum = __riscv_vwredsumu_vs_u16m4_u32m1(pixel_squared_vector2, temp_sqsum, vl);
//                     temp_sqsum = __riscv_vwredsumu_vs_u16m4_u32m1(pixel_squared_vector3, temp_sqsum, vl);
//                     temp_sqsum = __riscv_vwredsumu_vs_u16m4_u32m1(pixel_squared_vector4, temp_sqsum, vl);
//                 }
//             }
//             // vuint64m1_t <- vuint16m1_t
//             vuint64m1_t temp_sum = __riscv_vreinterpret_u64m1(vec_s);
//             // vec_sum[0] = sum( temp_sum , vec_sum)
//             vec_sum = __riscv_vadd_vv_u64m1(temp_sum, vec_sum, vl);
//             if (stddev_val) {
//                 temp_sum = __riscv_vreinterpret_u64m1(temp_sqsum);
//                 vec_sqsum = __riscv_vadd_vv_u64m1(temp_sum, vec_sqsum, vl);
//             }
//             total_count += vl;
//             j += vl;
//         }
//     }
//     for ( ; i < height; ++i) {
//         const uchar* src_row = src_data + i * src_step;
//         const uchar* mask_row = mask ? (mask + i * mask_step) : nullptr;
//         for ( ; j < width; ) {
//             vl = __riscv_vsetvl_e8m1(width-j); // tail elements
//             // Load src[row][i .. i+vl]
//             vuint8m2_t pixel_vector = __riscv_vle8_v_u8m2(src_row + j, vl);
//             vuint16m4_t pixel_squared_vector;
//             if(mask) {
//                 // Load mask[row][i .. i+vl]
//                 vbool4_t mask_vector = __riscv_vlm_v_b4(mask_row + j, vl);
//                 // vec_s[0] <- sum(vec_u16m1_zero[0] , pixel_vector[*]) , if not masked
//                 vec_s = __riscv_vwredsumu_vs_u8m2_u16m1_m(mask_vector, pixel_vector, vec_s, vl);
//                 if(stddev_val) {
//                     // pixel_squared_vector[*] = pixel_vector[*] * pixel_vector[*] , if not masked
//                     pixel_squared_vector = __riscv_vwmulu_vv_u16m4_m(mask_vector, pixel_vector, pixel_vector, vl);
//                     // sqsum[0] <- sum(vec_u32m2_zero[0] , pixel_squared_vector[*]) , if not masked
//                     temp_sqsum = __riscv_vwredsumu_vs_u16m4_u32m1_m(mask_vector, pixel_squared_vector, temp_sqsum, vl);
//                 }
//             }
//             else {
//                 // vec_s[0] <- sum(vec_u16m1_zero[0] , pixel_vector[*])
//                 vec_s = __riscv_vwredsumu_vs_u8m2_u16m1(pixel_vector, vec_s, vl);
//                 if(stddev_val) {
//                     // pixel_squared_vector[*] = pixel_vector[*] * pixel_vector[*]
//                     pixel_squared_vector = __riscv_vwmulu_vv_u16m4(pixel_vector, pixel_vector, vl);
//                     // sqsum[0] <- sum(vec_u32m2_zero[0] , pixel_squared_vector[*])
//                     temp_sqsum = __riscv_vwredsumu_vs_u16m4_u32m1(pixel_squared_vector, temp_sqsum, vl);
//                 }
//             }
//             // vuint64m1_t <- vuint16m1_t
//             vuint64m1_t temp_sum = __riscv_vreinterpret_u64m1(vec_s);
//             // vec_sum[0] = sum( temp_sum , vec_sum)
//             vec_sum = __riscv_vadd_vv_u64m1(temp_sum, vec_sum, vl);
//             if (stddev_val) {
//                 temp_sum = __riscv_vreinterpret_u64m1(temp_sqsum);
//                 vec_sqsum = __riscv_vadd_vv_u64m1(temp_sum, vec_sqsum, vl);
//             }
//             total_count += vl;
//             j += vl;
//         }
//     }
//     if (total_count == 0)
//     {
//         if (mean_val) *mean_val = 0.0;
//         if (stddev_val) *stddev_val = 0.0;
//         return CV_HAL_ERROR_OK;
//     }
//     // Return values
//     vfloat64m1_t float_sum = __riscv_vreinterpret_f64m1(vec_sum);
//     double sum = __riscv_vfmv_f_s_f64m1_f64(float_sum);
//     double mean = sum / total_count;
//     if (mean_val) {
//         *mean_val = mean;
//     }
//     if (stddev_val) {
//         vfloat64m1_t float_sqsum = __riscv_vreinterpret_f64m1(vec_sqsum);
//         double sqsum = __riscv_vfmv_f_s_f64m1_f64(float_sqsum);
//         double variance = std::max((sqsum / total_count) - (mean * mean), 0.0);
//         double stddev = std::sqrt(variance);
//         *stddev_val = stddev;
//     }
//     return CV_HAL_ERROR_OK;
// }

// // -----------------  meanStdDev_8u1c vector final plus version  -----------------
// // Brief: abstract some steps
// template<typename T>
// std::vector<T*> getParamsPool(const T* data, size_t step, size_t index, size_t loop, std::function<T*(const T*, size_t, size_t, size_t)> func) {
//     std::vector<T*> pool = std::vector<T*>(loop);
//     for(size_t i=0; i<loop; ++i) {
//         pool[i] = func(data, step, index, i);
//     }
//     return pool;
// }
// template<typename T>
// T* getSrcRow(const T* data, size_t step, size_t index, size_t loop) {
//
//     return const_cast<T*>(data + (index + loop) * step);
// }
//
// template<typename T>
// T* getMaskRow(const T* data, size_t step, size_t index, size_t loop) {
//     return data ? const_cast<T*>(data + (index + loop) * step) : nullptr;
// }
// inline int meanStdDev_8u1c(const uchar* src_data, size_t src_step, int width, int height,
//                             double* mean_val, double* stddev_val, uchar* mask, size_t mask_step) {
//     // initialize variables
//     size_t loop_unroll = 4;
//     size_t loop_rem = height % loop_unroll; 
//     size_t total_count = 0;
//     size_t vlmax = __riscv_vsetvlmax_e8m1();
//     vuint64m1_t vec_sum = __riscv_vmv_v_x_u64m1(0, vlmax);
//     vuint64m1_t vec_sqsum = __riscv_vmv_v_x_u64m1(0, vlmax);
//     vuint16m1_t vec_s = __riscv_vmv_v_x_u16m1(0, vlmax);
//     vuint32m1_t temp_sqsum = __riscv_vmv_v_x_u32m1(0, vlmax);
//     int j = 0;
//     int vl = __riscv_vsetvl_e8m1(width-j);
//     size_t i=0;
//     for ( ; i < height - loop_rem; i+=4) {
//         std::vector<uchar*> src_row_pool = getParamsPool(src_data, src_step, i, loop_unroll, getSrcRow<uchar>);
//         std::vector<uchar*> mask_row_pool = getParamsPool(mask, mask_step, i, loop_unroll, getMaskRow<uchar>);
//         for ( ; j < width; ) {
//             vl = __riscv_vsetvl_e8m1(width-j); // tail elements
//             std::vector<vuint16m2_t> pixel_squared_vector_pool(loop_unroll);
//             // Load src[row][i .. i+vl]
//             std::vector<vuint8m1_t> pixel_vector_pool(loop_unroll);
//             for(size_t k; k<loop_unroll; ++k) {
//                 pixel_vector_pool[k] = __riscv_vle8_v_u8m1(src_row_pool[k] + j, vl);
//             }
//             if(mask) {
//                 // Load mask[row][i .. i+vl]
//                 std::vector<vbool8_t> mask_vector_pool(loop_unroll);
//                 for(size_t k; k<loop_unroll; ++k) {
//                     mask_vector_pool[k] = __riscv_vlm_v_b8(mask_row_pool[k] + j, vl);
//                 }
//                 // vec_s[0] <- sum(vec_s[0] , pixel_vector[*]) , if not masked
//                 for(size_t k; k<loop_unroll; ++k) {
//                     vec_s = __riscv_vwredsumu_vs_u8m1_u16m1_m(mask_vector_pool[k], pixel_vector_pool[k], vec_s, vl);
//                 }
//                 if(stddev_val) {
//                     // pixel_squared_vector[*] = pixel_vector[*] * pixel_vector[*] , if not masked
//                     for(size_t k; k<loop_unroll; ++k) {
//                         pixel_squared_vector_pool[k] = __riscv_vwmulu_vv_u16m2_m(mask_vector_pool[k], pixel_vector_pool[k], pixel_vector_pool[k], vl);
//                     }
//                     // sqsum[0] <- sum(temp_sqsum[0] , pixel_squared_vector[*]) , if not masked
//                     for(size_t k; k<loop_unroll; ++k) {
//                         temp_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1_m(mask_vector_pool[k], pixel_squared_vector_pool[k], temp_sqsum, vl);
//                     }
//                 }
//             }
//             else {
//                 // vec_s[0] <- sum(vec_u16m1_zero[0] , pixel_vector[*])
//                 for(size_t k; k<loop_unroll; ++k) {
//                     vec_s = __riscv_vwredsumu_vs_u8m1_u16m1(pixel_vector_pool[k], vec_s, vl);
//                 }
//                 if(stddev_val) {
//                     // pixel_squared_vector[*] = pixel_vector[*] * pixel_vector[*]
//                     for(size_t k; k<loop_unroll; ++k) {
//                         pixel_squared_vector_pool[k] = __riscv_vwmulu_vv_u16m2(pixel_vector_pool[k], pixel_vector_pool[k], vl);
//                     }
//                     // sqsum[0] <- sum(vec_u32m2_zero[0] , pixel_squared_vector[*])
//                     for(size_t k; k<loop_unroll; ++k) {
//                         temp_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1(pixel_squared_vector_pool[k], temp_sqsum, vl);
//                     }
//                 }
//             }
//             // vuint64m1_t <- vuint16m1_t
//             vuint64m1_t temp_sum = __riscv_vreinterpret_u64m1(vec_s);
//             // vec_sum[0] = sum( temp_sum , vec_sum)
//             vec_sum = __riscv_vadd_vv_u64m1(temp_sum, vec_sum, vl);
//             if (stddev_val) {
//                 temp_sum = __riscv_vreinterpret_u64m1(temp_sqsum);
//                 vec_sqsum = __riscv_vadd_vv_u64m1(temp_sum, vec_sqsum, vl);
//             }
//             total_count += vl;
//             j += vl;
//         }
//     }
//     for ( ; i < height; ++i) {
//         const uchar* src_row = src_data + i * src_step;
//         const uchar* mask_row = mask ? (mask + i * mask_step) : nullptr;
//         for ( ; j < width; ) {
//             vl = __riscv_vsetvl_e8m1(width-j); // tail elements
//             // Load src[row][i .. i+vl]
//             vuint8m1_t pixel_vector = __riscv_vle8_v_u8m1(src_row + j, vl);
//             vuint16m2_t pixel_squared_vector;
//             if(mask) {
//                 // Load mask[row][i .. i+vl]
//                 vbool8_t mask_vector = __riscv_vlm_v_b8(mask_row + j, vl);
//                 // vec_s[0] <- sum(vec_u16m1_zero[0] , pixel_vector[*]) , if not masked
//                 vec_s = __riscv_vwredsumu_vs_u8m1_u16m1_m(mask_vector, pixel_vector, vec_s, vl);
//                 if(stddev_val) {
//                     // pixel_squared_vector[*] = pixel_vector[*] * pixel_vector[*] , if not masked
//                     pixel_squared_vector = __riscv_vwmulu_vv_u16m2_m(mask_vector, pixel_vector, pixel_vector, vl);
//                     // sqsum[0] <- sum(vec_u32m2_zero[0] , pixel_squared_vector[*]) , if not masked
//                     temp_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1_m(mask_vector, pixel_squared_vector, temp_sqsum, vl);
//                 }
//             }
//             else {
//                 // vec_s[0] <- sum(vec_u16m1_zero[0] , pixel_vector[*])
//                 vec_s = __riscv_vwredsumu_vs_u8m1_u16m1(pixel_vector, vec_s, vl);
//                 if(stddev_val) {
//                     // pixel_squared_vector[*] = pixel_vector[*] * pixel_vector[*]
//                     pixel_squared_vector = __riscv_vwmulu_vv_u16m2(pixel_vector, pixel_vector, vl);
//                     // sqsum[0] <- sum(vec_u32m2_zero[0] , pixel_squared_vector[*])
//                     temp_sqsum = __riscv_vwredsumu_vs_u16m2_u32m1(pixel_squared_vector, temp_sqsum, vl);
//                 }
//             }
//             // vuint64m1_t <- vuint16m1_t
//             vuint64m1_t temp_sum = __riscv_vreinterpret_u64m1(vec_s);
//             // vec_sum[0] = sum( temp_sum , vec_sum)
//             vec_sum = __riscv_vadd_vv_u64m1(temp_sum, vec_sum, vl);
//             if (stddev_val) {
//                 temp_sum = __riscv_vreinterpret_u64m1(temp_sqsum);
//                 vec_sqsum = __riscv_vadd_vv_u64m1(temp_sum, vec_sqsum, vl);
//             }
//             total_count += vl;
//             j += vl;
//         }
//     }
//     if (total_count == 0)
//     {
//         if (mean_val) *mean_val = 0.0;
//         if (stddev_val) *stddev_val = 0.0;
//         return CV_HAL_ERROR_OK;
//     }
//     // Return values
//     vfloat64m1_t float_sum = __riscv_vreinterpret_f64m1(vec_sum);
//     double sum = __riscv_vfmv_f_s_f64m1_f64(float_sum);
//     double mean = sum / total_count;
//     if (mean_val) {
//         *mean_val = mean;
//     }
//     if (stddev_val) {
//         vfloat64m1_t float_sqsum = __riscv_vreinterpret_f64m1(vec_sqsum);
//         double sqsum = __riscv_vfmv_f_s_f64m1_f64(float_sqsum);
//         double variance = std::max((sqsum / total_count) - (mean * mean), 0.0);
//         double stddev = std::sqrt(variance);
//         *stddev_val = stddev;
//     }
//     return CV_HAL_ERROR_OK;
// }

}}
#endif


