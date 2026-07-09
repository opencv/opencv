/*
 * simoncatbot-opencv RPP HAL - Imgproc Module
 *
 * AMD ROCm Performance Primitives (RPP) HAL for OpenCV 5.x imgproc
 */

#ifndef __RPP_HAL_IMGPROC_HPP__
#define __RPP_HAL_IMGPROC_HPP__

#include <opencv2/core/base.hpp>

#ifdef __cplusplus
extern "C" {
#endif

// =========================================================================
// FILTERING
// =========================================================================

int rpp_hal_boxFilter(const uchar* src_data, size_t src_step,
                      uchar* dst_data, size_t dst_step,
                      int width, int height, int src_depth, int dst_depth, int cn,
                      int margin_left, int margin_top, int margin_right, int margin_bottom,
                      size_t ksize_width, size_t ksize_height,
                      int anchor_x, int anchor_y,
                      bool normalize, int border_type);

int rpp_hal_gaussianBlur(const uchar* src_data, size_t src_step,
                         uchar* dst_data, size_t dst_step,
                         int width, int height, int depth, int cn,
                         size_t margin_left, size_t margin_top, size_t margin_right, size_t margin_bottom,
                         size_t ksize_width, size_t ksize_height,
                         double sigmaX, double sigmaY, int border_type);

int rpp_hal_medianBlur(const uchar* src_data, size_t src_step,
                       uchar* dst_data, size_t dst_step,
                       int width, int height, int depth, int cn, int ksize);

// =========================================================================
// DERIVATIVES — no RPP GPU APIs, always NOT_IMPLEMENTED
// =========================================================================

int rpp_hal_sobel(const uchar* src_data, size_t src_step,
                  uchar* dst_data, size_t dst_step,
                  int width, int height, int src_depth, int dst_depth, int cn,
                  int margin_left, int margin_top, int margin_right, int margin_bottom,
                  int dx, int dy, int ksize, double scale, double delta, int border_type);

// =========================================================================
// FEATURES — no RPP GPU APIs, always NOT_IMPLEMENTED
// =========================================================================

int rpp_hal_canny(const uchar* src_data, size_t src_step,
                  uchar* dst_data, size_t dst_step,
                  int width, int height, int cn,
                  double lowThreshold, double highThreshold, int ksize, bool L2gradient);

// =========================================================================
// MORPHOLOGY — NOT IMPLEMENTED (complex HAL interface, simple box-kernel only)
// =========================================================================

// We don't hook cv_hal_morph_stateless/cv_hal_morphInit/cv_hal_morph/cv_hal_morphFree
// because RPP only supports simple box-kernel erode/dilate, and the HAL interface
// is complex (cvhalFilter2D context, ROI offsets, etc.).
// OpenCV will fall back to CPU for morphology.

// =========================================================================
// GEOMETRY
// =========================================================================

int rpp_hal_resize(int src_type,
                   const uchar* src_data, size_t src_step,
                   int src_width, int src_height,
                   uchar* dst_data, size_t dst_step,
                   int dst_width, int dst_height,
                   double inv_scale_x, double inv_scale_y, int interpolation);

int rpp_hal_warpAffine(int src_type,
                       const uchar* src_data, size_t src_step,
                       int src_width, int src_height,
                       uchar* dst_data, size_t dst_step,
                       int dst_width, int dst_height,
                       const double M[6], int interpolation,
                       int borderType, const double borderValue[4]);

int rpp_hal_warpPerspective(int src_type,
                            const uchar* src_data, size_t src_step,
                            int src_width, int src_height,
                            uchar* dst_data, size_t dst_step,
                            int dst_width, int dst_height,
                            const double M[9], int interpolation,
                            int borderType, const double borderValue[4]);

int rpp_hal_flip(int src_type,
                 const uchar* src_data, size_t src_step,
                 int src_width, int src_height,
                 uchar* dst_data, size_t dst_step,
                 int flip_mode);

// =========================================================================
// COLOR CONVERSIONS — no RPP GPU APIs, always NOT_IMPLEMENTED
// =========================================================================

int rpp_hal_cvtBGRtoBGR(const uchar* src_data, size_t src_step,
                        uchar* dst_data, size_t dst_step,
                        int width, int height, int depth,
                        int scn, int dcn, bool swapBlue);

int rpp_hal_cvtBGRtoGray(const uchar* src_data, size_t src_step,
                         uchar* dst_data, size_t dst_step,
                         int width, int height, int depth,
                         int scn, bool swapBlue);

int rpp_hal_cvtGraytoBGR(const uchar* src_data, size_t src_step,
                         uchar* dst_data, size_t dst_step,
                         int width, int height, int depth, int dcn);

int rpp_hal_cvtBGRtoHSV(const uchar* src_data, size_t src_step,
                        uchar* dst_data, size_t dst_step,
                        int width, int height, int depth,
                        int scn, bool swapBlue, bool isFullRange, bool isHSV);

int rpp_hal_cvtHSVtoBGR(const uchar* src_data, size_t src_step,
                        uchar* dst_data, size_t dst_step,
                        int width, int height, int depth,
                        int dcn, bool swapBlue, bool isFullRange, bool isHSV);

#ifdef __cplusplus
}
#endif

// =========================================================================
// Register HAL hooks with OpenCV
// =========================================================================

#undef cv_hal_boxFilter
#define cv_hal_boxFilter rpp_hal_boxFilter

#undef cv_hal_gaussianBlur
#define cv_hal_gaussianBlur rpp_hal_gaussianBlur

#undef cv_hal_medianBlur
#define cv_hal_medianBlur rpp_hal_medianBlur

#undef cv_hal_sobel
#define cv_hal_sobel rpp_hal_sobel

#undef cv_hal_canny
#define cv_hal_canny rpp_hal_canny

#undef cv_hal_resize
#define cv_hal_resize rpp_hal_resize

#undef cv_hal_warpAffine
#define cv_hal_warpAffine rpp_hal_warpAffine

#undef cv_hal_warpPerspective
#define cv_hal_warpPerspective rpp_hal_warpPerspective

#undef cv_hal_flip
#define cv_hal_flip rpp_hal_flip

#undef cv_hal_cvtBGRtoBGR
#define cv_hal_cvtBGRtoBGR rpp_hal_cvtBGRtoBGR
#undef cv_hal_cvtBGRtoGray
#define cv_hal_cvtBGRtoGray rpp_hal_cvtBGRtoGray
#undef cv_hal_cvtGraytoBGR
#define cv_hal_cvtGraytoBGR rpp_hal_cvtGraytoBGR
#undef cv_hal_cvtBGRtoHSV
#define cv_hal_cvtBGRtoHSV rpp_hal_cvtBGRtoHSV
#undef cv_hal_cvtHSVtoBGR
#define cv_hal_cvtHSVtoBGR rpp_hal_cvtHSVtoBGR

#endif // __RPP_HAL_IMGPROC_HPP__
