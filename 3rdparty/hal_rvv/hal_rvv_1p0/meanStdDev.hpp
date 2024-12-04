// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_MEANSTDDEV_HPP_INCLUDED
#define OPENCV_HAL_RVV_MEANSTDDEV_HPP_INCLUDED
#include <riscv_vector.h>
namespace cv { namespace cv_hal_rvv {
#undef cv_hal_meanStdDev
#define cv_hal_meanStdDev cv::cv_hal_rvv::meanStdDev

inline int meanStdDev_8u1c(const uchar* src_data, size_t src_step, int width, int height,
                            double* mean_val, double* stddev_val, uchar* mask, size_t mask_step);

inline int meanStdDev(const uchar* src_data, size_t src_step, int width, int height,
                             int src_type, double* mean_val, double* stddev_val, uchar* mask, size_t mask_step) {
    if(src_type == CV_8UC1) 
        return meanStdDev_8u1c(src_data, src_step, width, height, mean_val, stddev_val, mask, mask_step);
    else return CV_HAL_ERROR_NOT_IMPLEMENTED;

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

//             // Load src[row][i .. i+vl]
//             vuint8m1_t pixel_vector = __riscv_vle8_v_u8m1(src_row + j, vl);
//             // Load mask[row][i .. i+vl]
//             vuint16m1_t vec_s;
//             vuint16m2_t pixel_squared_vector;
//             vuint32m1_t temp_sqsum;
//             if(mask_row) {
//                 vbool8_t mask_vector = __riscv_vlm_v_b8(mask_row + j, vl);

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
}}
#endif