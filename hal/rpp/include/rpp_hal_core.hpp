/*
 * simoncatbot-opencv RPP HAL - Core Module
 *
 * AMD ROCm Performance Primitives (RPP) HAL for OpenCV 5.x
 * Provides GPU/CPU core operations via RPP backend.
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

// Add
int rpp_hal_add8u(const uchar* src1_data, size_t src1_step,
                  const uchar* src2_data, size_t src2_step,
                  uchar* dst_data, size_t dst_step,
                  int width, int height);
int rpp_hal_add16s(const short* src1_data, size_t src1_step,
                   const short* src2_data, size_t src2_step,
                   short* dst_data, size_t dst_step,
                   int width, int height);
int rpp_hal_add32f(const float* src1_data, size_t src1_step,
                   const float* src2_data, size_t src2_step,
                   float* dst_data, size_t dst_step,
                   int width, int height);

// Subtract
int rpp_hal_sub8u(const uchar* src1_data, size_t src1_step,
                  const uchar* src2_data, size_t src2_step,
                  uchar* dst_data, size_t dst_step,
                  int width, int height);
int rpp_hal_sub16s(const short* src1_data, size_t src1_step,
                   const short* src2_data, size_t src2_step,
                   short* dst_data, size_t dst_step,
                   int width, int height);
int rpp_hal_sub32f(const float* src1_data, size_t src1_step,
                   const float* src2_data, size_t src2_step,
                   float* dst_data, size_t dst_step,
                   int width, int height);

// Multiply
int rpp_hal_mul8u(const uchar* src1_data, size_t src1_step,
                  const uchar* src2_data, size_t src2_step,
                  uchar* dst_data, size_t dst_step,
                  int width, int height, double scale);
int rpp_hal_mul16s(const short* src1_data, size_t src1_step,
                   const short* src2_data, size_t src2_step,
                   short* dst_data, size_t dst_step,
                   int width, int height, double scale);
int rpp_hal_mul32f(const float* src1_data, size_t src1_step,
                   const float* src2_data, size_t src2_step,
                   float* dst_data, size_t dst_step,
                   int width, int height, double scale);

// Divide / reciprocal
int rpp_hal_div32f(const float* src1_data, size_t src1_step,
                   const float* src2_data, size_t src2_step,
                   float* dst_data, size_t dst_step,
                   int width, int height, double scale);
int rpp_hal_recip32f(const float* src_data, size_t src_step,
                     float* dst_data, size_t dst_step,
                     int width, int height, double scale);

// AddWeighted
int rpp_hal_addWeighted8u(const uchar* src1_data, size_t src1_step,
                          const uchar* src2_data, size_t src2_step,
                          uchar* dst_data, size_t dst_step,
                          int width, int height,
                          const double scalars[3]);
int rpp_hal_addWeighted32f(const float* src1_data, size_t src1_step,
                           const float* src2_data, size_t src2_step,
                           float* dst_data, size_t dst_step,
                           int width, int height,
                           const double scalars[3]);

// Type conversions
int rpp_hal_cvt8u16s(const uchar* src_data, size_t src_step,
                     short* dst_data, size_t dst_step,
                     int width, int height);
int rpp_hal_cvt16s8u(const short* src_data, size_t src_step,
                     uchar* dst_data, size_t dst_step,
                     int width, int height);
int rpp_hal_cvt16s32f(const short* src_data, size_t src_step,
                      float* dst_data, size_t dst_step,
                      int width, int height, double scale);
int rpp_hal_cvt32f16s(const float* src_data, size_t src_step,
                      short* dst_data, size_t dst_step,
                      int width, int height, double scale);

// Abs
int rpp_hal_abs8u(const uchar* src_data, size_t src_step,
                  uchar* dst_data, size_t dst_step,
                  int width, int height);
int rpp_hal_abs16s(const short* src_data, size_t src_step,
                   short* dst_data, size_t dst_step,
                   int width, int height);
int rpp_hal_abs32f(const float* src_data, size_t src_step,
                   float* dst_data, size_t dst_step,
                   int width, int height);

// Compare
int rpp_hal_cmp8u(const uchar* src1_data, size_t src1_step,
                  const uchar* src2_data, size_t src2_step,
                  uchar* dst_data, size_t dst_step,
                  int width, int height, int cmpop);
int rpp_hal_cmp16s(const short* src1_data, size_t src1_step,
                   const short* src2_data, size_t src2_step,
                   uchar* dst_data, size_t dst_step,
                   int width, int height, int cmpop);
int rpp_hal_cmp32f(const float* src1_data, size_t src1_step,
                   const float* src2_data, size_t src2_step,
                   uchar* dst_data, size_t dst_step,
                   int width, int height, int cmpop);
int rpp_hal_cmp64f(const double* src1_data, size_t src1_step,
                   const double* src2_data, size_t src2_step,
                   uchar* dst_data, size_t dst_step,
                   int width, int height, int cmpop);

// minMaxLoc (OpenCV HAL name is minMaxIdx)
int rpp_hal_minMaxIdx8u(const uchar* src_data, size_t src_step,
                          double* minVal, double* maxVal,
                          int* minIdx, int* maxIdx,
                          int width, int height, int cn);
int rpp_hal_minMaxIdx16s(const short* src_data, size_t src_step,
                           double* minVal, double* maxVal,
                           int* minIdx, int* maxIdx,
                           int width, int height, int cn);
int rpp_hal_minMaxIdx32f(const float* src_data, size_t src_step,
                           double* minVal, double* maxVal,
                           int* minIdx, int* maxIdx,
                           int width, int height, int cn);
int rpp_hal_minMaxIdx64f(const double* src_data, size_t src_step,
                           double* minVal, double* maxVal,
                           int* minIdx, int* maxIdx,
                           int width, int height, int cn);

// countNonZero
int rpp_hal_countNonZero8u(const uchar* src_data, int len);
int rpp_hal_countNonZero16s(const short* src_data, int len);
int rpp_hal_countNonZero32f(const float* src_data, int len);
int rpp_hal_countNonZero64f(const double* src_data, int len);

// dotProduct
int rpp_hal_dotProduct8u(const uchar* src1_data, size_t src1_step,
                         const uchar* src2_data, size_t src2_step,
                         int width, int height, double* result);
int rpp_hal_dotProduct8s(const char* src1_data, size_t src1_step,
                         const char* src2_data, size_t src2_step,
                         int width, int height, double* result);
int rpp_hal_dotProduct16u(const ushort* src1_data, size_t src1_step,
                          const ushort* src2_data, size_t src2_step,
                          int width, int height, double* result);
int rpp_hal_dotProduct32f(const float* src1_data, size_t src1_step,
                          const float* src2_data, size_t src2_step,
                          int width, int height, double* result);
int rpp_hal_dotProduct64f(const double* src1_data, size_t src1_step,
                          const double* src2_data, size_t src2_step,
                          int width, int height, double* result);

// meanStdDev
int rpp_hal_meanStdDev8u(const uchar* src_data, size_t src_step,
                         int width, int height,
                         double* meanVal, double* stdDevVal,
                         uchar* mask, size_t maskStep);
int rpp_hal_meanStdDev16u(const ushort* src_data, size_t src_step,
                          int width, int height,
                          double* meanVal, double* stdDevVal,
                          uchar* mask, size_t maskStep);
int rpp_hal_meanStdDev32f(const float* src_data, size_t src_step,
                          int width, int height,
                          double* meanVal, double* stdDevVal,
                          uchar* mask, size_t maskStep);
int rpp_hal_meanStdDev64f(const double* src_data, size_t src_step,
                          int width, int height,
                          double* meanVal, double* stdDevVal,
                          uchar* mask, size_t maskStep);

// integral
int rpp_hal_integral8u(const uchar* src_data, size_t src_step,
                     double* dst_data, size_t dst_step,
                     int width, int height, int cn);
int rpp_hal_integral32f(const float* src_data, size_t src_step,
                        double* dst_data, size_t dst_step,
                        int width, int height, int cn);
int rpp_hal_integral32s(const int* src_data, size_t src_step,
                        double* dst_data, size_t dst_step,
                        int width, int height, int cn);

// cvtColor stubs
int rpp_hal_cvtBGRtoGray8u(const uchar* src_data, size_t src_step,
                           uchar* dst_data, size_t dst_step,
                           int width, int height, int cn, int blueIdx);
int rpp_hal_cvtBGRtoGray16u(const ushort* src_data, size_t src_step,
                            ushort* dst_data, size_t dst_step,
                            int width, int height, int cn, int blueIdx);
int rpp_hal_cvtBGRtoGray32f(const float* src_data, size_t src_step,
                            float* dst_data, size_t dst_step,
                            int width, int height, int cn, int blueIdx);
int rpp_hal_cvtGraytoBGR8u(const uchar* src_data, size_t src_step,
                           uchar* dst_data, size_t dst_step,
                           int width, int height, int cn);

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
#undef cv_hal_add16s
#define cv_hal_add16s rpp_hal_add16s
#undef cv_hal_add32f
#define cv_hal_add32f rpp_hal_add32f

#undef cv_hal_sub8u
#define cv_hal_sub8u rpp_hal_sub8u
#undef cv_hal_sub16s
#define cv_hal_sub16s rpp_hal_sub16s
#undef cv_hal_sub32f
#define cv_hal_sub32f rpp_hal_sub32f

#undef cv_hal_mul8u
#define cv_hal_mul8u rpp_hal_mul8u
#undef cv_hal_mul16s
#define cv_hal_mul16s rpp_hal_mul16s
#undef cv_hal_mul32f
#define cv_hal_mul32f rpp_hal_mul32f

#undef cv_hal_div32f
#define cv_hal_div32f rpp_hal_div32f
#undef cv_hal_recip32f
#define cv_hal_recip32f rpp_hal_recip32f

#undef cv_hal_addWeighted8u
#define cv_hal_addWeighted8u rpp_hal_addWeighted8u
#undef cv_hal_addWeighted32f
#define cv_hal_addWeighted32f rpp_hal_addWeighted32f

#undef cv_hal_cvt8u16s
#define cv_hal_cvt8u16s rpp_hal_cvt8u16s
#undef cv_hal_cvt16s8u
#define cv_hal_cvt16s8u rpp_hal_cvt16s8u
#undef cv_hal_cvt16s32f
#define cv_hal_cvt16s32f rpp_hal_cvt16s32f
#undef cv_hal_cvt32f16s
#define cv_hal_cvt32f16s rpp_hal_cvt32f16s

#undef cv_hal_abs8u
#define cv_hal_abs8u rpp_hal_abs8u
#undef cv_hal_abs16s
#define cv_hal_abs16s rpp_hal_abs16s
#undef cv_hal_abs32f
#define cv_hal_abs32f rpp_hal_abs32f

#undef cv_hal_cmp8u
#define cv_hal_cmp8u rpp_hal_cmp8u
#undef cv_hal_cmp16s
#define cv_hal_cmp16s rpp_hal_cmp16s
#undef cv_hal_cmp32f
#define cv_hal_cmp32f rpp_hal_cmp32f
#undef cv_hal_cmp64f
#define cv_hal_cmp64f rpp_hal_cmp64f

#undef cv_hal_minMaxIdx8u
#define cv_hal_minMaxIdx8u rpp_hal_minMaxIdx8u
#undef cv_hal_minMaxIdx16s
#define cv_hal_minMaxIdx16s rpp_hal_minMaxIdx16s
#undef cv_hal_minMaxIdx32f
#define cv_hal_minMaxIdx32f rpp_hal_minMaxIdx32f
#undef cv_hal_minMaxIdx64f
#define cv_hal_minMaxIdx64f rpp_hal_minMaxIdx64f

#undef cv_hal_countNonZero8u
#define cv_hal_countNonZero8u rpp_hal_countNonZero8u
#undef cv_hal_countNonZero16s
#define cv_hal_countNonZero16s rpp_hal_countNonZero16s
#undef cv_hal_countNonZero32f
#define cv_hal_countNonZero32f rpp_hal_countNonZero32f
#undef cv_hal_countNonZero64f
#define cv_hal_countNonZero64f rpp_hal_countNonZero64f

#undef cv_hal_dotProduct8u
#define cv_hal_dotProduct8u rpp_hal_dotProduct8u
#undef cv_hal_dotProduct8s
#define cv_hal_dotProduct8s rpp_hal_dotProduct8s
#undef cv_hal_dotProduct16u
#define cv_hal_dotProduct16u rpp_hal_dotProduct16u
#undef cv_hal_dotProduct32f
#define cv_hal_dotProduct32f rpp_hal_dotProduct32f
#undef cv_hal_dotProduct64f
#define cv_hal_dotProduct64f rpp_hal_dotProduct64f

#undef cv_hal_meanStdDev8u
#define cv_hal_meanStdDev8u rpp_hal_meanStdDev8u
#undef cv_hal_meanStdDev16u
#define cv_hal_meanStdDev16u rpp_hal_meanStdDev16u
#undef cv_hal_meanStdDev32f
#define cv_hal_meanStdDev32f rpp_hal_meanStdDev32f
#undef cv_hal_meanStdDev64f
#define cv_hal_meanStdDev64f rpp_hal_meanStdDev64f

#undef cv_hal_integral8u
#define cv_hal_integral8u rpp_hal_integral8u
#undef cv_hal_integral32f
#define cv_hal_integral32f rpp_hal_integral32f
#undef cv_hal_integral32s
#define cv_hal_integral32s rpp_hal_integral32s

#undef cv_hal_cvtBGRtoGray8u
#define cv_hal_cvtBGRtoGray8u rpp_hal_cvtBGRtoGray8u
#undef cv_hal_cvtBGRtoGray16u
#define cv_hal_cvtBGRtoGray16u rpp_hal_cvtBGRtoGray16u
#undef cv_hal_cvtBGRtoGray32f
#define cv_hal_cvtBGRtoGray32f rpp_hal_cvtBGRtoGray32f
#undef cv_hal_cvtGraytoBGR8u
#define cv_hal_cvtGraytoBGR8u rpp_hal_cvtGraytoBGR8u

#undef cv_hal_magnitude32f
#define cv_hal_magnitude32f rpp_hal_magnitude32f
#undef cv_hal_magnitude64f
#define cv_hal_magnitude64f rpp_hal_magnitude64f

#endif // __RPP_HAL_CORE_HPP__
