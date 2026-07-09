/*
 * simoncatbot-opencv RPP HAL - Core Module
 *
 * AMD ROCm Performance Primitives (RPP) HAL for OpenCV 5.x
 * Provides GPU-accelerated core operations via HIP backend
 */

#ifndef __RPP_HAL_CORE_HPP__
#define __RPP_HAL_CORE_HPP__

#include <opencv2/core/base.hpp>

#ifdef __cplusplus
extern "C" {
#endif

// =========================================================================
// RPP HAL Core Function Declarations
// Must match OpenCV hal interface signatures exactly.
// =========================================================================

// Bitwise operations
int rpp_hal_and8u(const uchar* src1_data, size_t src1_step,
                  const uchar* src2_data, size_t src2_step,
                  uchar* dst_data, size_t dst_step,
                  int width, int height);
int rpp_hal_or8u(const uchar* src1_data, size_t src1_step,
                 const uchar* src2_data, size_t src2_step,
                 uchar* dst_data, size_t dst_step,
                 int width, int height);
int rpp_hal_xor8u(const uchar* src1_data, size_t src1_step,
                  const uchar* src2_data, size_t src2_step,
                  uchar* dst_data, size_t dst_step,
                  int width, int height);
int rpp_hal_not8u(const uchar* src_data, size_t src_step,
                  uchar* dst_data, size_t dst_step,
                  int width, int height);

// Add (returns NOT_IMPLEMENTED until custom HIP kernel added)
int rpp_hal_add8u(const uchar* src1_data, size_t src1_step,
                  const uchar* src2_data, size_t src2_step,
                  uchar* dst_data, size_t dst_step,
                  int width, int height);
int rpp_hal_add32f(const float* src1_data, size_t src1_step,
                   const float* src2_data, size_t src2_step,
                   float* dst_data, size_t dst_step,
                   int width, int height);

// Multiply (returns NOT_IMPLEMENTED until custom HIP kernel added)
int rpp_hal_mul8u(const uchar* src1_data, size_t src1_step,
                  const uchar* src2_data, size_t src2_step,
                  uchar* dst_data, size_t dst_step,
                  int width, int height, double scale);
int rpp_hal_mul32f(const float* src1_data, size_t src1_step,
                   const float* src2_data, size_t src2_step,
                   float* dst_data, size_t dst_step,
                   int width, int height, double scale);

// LUT
int rpp_hal_lut(const uchar* src_data, size_t src_step,
                int width, int height, int cn,
                const uchar* lut_data, int lut_cn,
                uchar* dst_data, size_t dst_step);

// Magnitude
int rpp_hal_magnitude32f(const float* x_data, const float* y_data,
                         float* dst_data, int len);
int rpp_hal_magnitude64f(const double* x_data, const double* y_data,
                         double* dst_data, int len);

#ifdef __cplusplus
}
#endif

// =========================================================================
// Register HAL hooks with OpenCV
// =========================================================================

#undef cv_hal_and8u
#define cv_hal_and8u rpp_hal_and8u
#undef cv_hal_or8u
#define cv_hal_or8u rpp_hal_or8u
#undef cv_hal_xor8u
#define cv_hal_xor8u rpp_hal_xor8u
#undef cv_hal_not8u
#define cv_hal_not8u rpp_hal_not8u

#undef cv_hal_add8u
#define cv_hal_add8u rpp_hal_add8u
#undef cv_hal_add32f
#define cv_hal_add32f rpp_hal_add32f

#undef cv_hal_mul8u
#define cv_hal_mul8u rpp_hal_mul8u
#undef cv_hal_mul32f
#define cv_hal_mul32f rpp_hal_mul32f


#undef cv_hal_magnitude32f
#define cv_hal_magnitude32f rpp_hal_magnitude32f
#undef cv_hal_magnitude64f
#define cv_hal_magnitude64f rpp_hal_magnitude64f

#endif // __RPP_HAL_CORE_HPP__
