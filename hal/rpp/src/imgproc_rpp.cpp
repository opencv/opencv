/**
 * simoncatbot-opencv RPP HAL - Imgproc Operations (Unified dispatch)
 *
 * Architecture:
 *   1. GPU: Uses RPP HIP backend with device memory upload/download
 *   2. CPU: Uses RPP HOST backend (no memory copies, direct pointer pass)
 *   3. Fallback: Returns CV_HAL_ERROR_NOT_IMPLEMENTED → OpenCV native
 *
 * NOTE: Currently only core bitwise ops have full GPU+CPU dispatch.
 * Imgproc functions are stubbed — they will be added with verified RPP signatures.
 */

#include "rpp_hal_imgproc.hpp"
#include "rpp_hal_utils.hpp"

using namespace cv::hal::rpp;

// =========================================================================
// FILTERING
// =========================================================================

extern "C" int rpp_hal_boxFilter(const uchar* src_data, size_t src_step,
                                 uchar* dst_data, size_t dst_step,
                                 int width, int height, int src_depth, int dst_depth, int cn,
                                 int margin_left, int margin_top, int margin_right, int margin_bottom,
                                 size_t ksize_width, size_t ksize_height,
                                 int anchor_x, int anchor_y,
                                 bool normalize, int border_type) {
    (void)src_data; (void)src_step; (void)dst_data; (void)dst_step;
    (void)width; (void)height; (void)src_depth; (void)dst_depth; (void)cn;
    (void)margin_left; (void)margin_top; (void)margin_right; (void)margin_bottom;
    (void)ksize_width; (void)ksize_height; (void)anchor_x; (void)anchor_y;
    (void)normalize; (void)border_type;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_gaussianBlur(const uchar* src_data, size_t src_step,
                                    uchar* dst_data, size_t dst_step,
                                    int width, int height, int depth, int cn,
                                    size_t margin_left, size_t margin_top, size_t margin_right, size_t margin_bottom,
                                    size_t ksize_width, size_t ksize_height,
                                    double sigmaX, double sigmaY, int border_type) {
    (void)src_data; (void)src_step; (void)dst_data; (void)dst_step;
    (void)width; (void)height; (void)depth; (void)cn;
    (void)margin_left; (void)margin_top; (void)margin_right; (void)margin_bottom;
    (void)ksize_width; (void)ksize_height; (void)sigmaX; (void)sigmaY; (void)border_type;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_medianBlur(const uchar* src_data, size_t src_step,
                                  uchar* dst_data, size_t dst_step,
                                  int width, int height, int depth, int cn, int ksize) {
    (void)src_data; (void)src_step; (void)dst_data; (void)dst_step;
    (void)width; (void)height; (void)depth; (void)cn; (void)ksize;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

// =========================================================================
// DERIVATIVES
// =========================================================================

extern "C" int rpp_hal_sobel(const uchar* src_data, size_t src_step,
                             uchar* dst_data, size_t dst_step,
                             int width, int height, int src_depth, int dst_depth, int cn,
                             int margin_left, int margin_top, int margin_right, int margin_bottom,
                             int dx, int dy, int ksize, double scale, double delta, int border_type) {
    (void)src_data; (void)src_step; (void)dst_data; (void)dst_step;
    (void)width; (void)height; (void)src_depth; (void)dst_depth; (void)cn;
    (void)margin_left; (void)margin_top; (void)margin_right; (void)margin_bottom;
    (void)dx; (void)dy; (void)ksize; (void)scale; (void)delta; (void)border_type;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

// =========================================================================
// FEATURES
// =========================================================================

extern "C" int rpp_hal_canny(const uchar* src_data, size_t src_step,
                             uchar* dst_data, size_t dst_step,
                             int width, int height, int cn,
                             double lowThreshold, double highThreshold, int ksize, bool L2gradient) {
    (void)src_data; (void)src_step; (void)dst_data; (void)dst_step;
    (void)width; (void)height; (void)cn;
    (void)lowThreshold; (void)highThreshold; (void)ksize; (void)L2gradient;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

// =========================================================================
// GEOMETRY
// =========================================================================

extern "C" int rpp_hal_resize(int src_type,
                              const uchar* src_data, size_t src_step,
                              int src_width, int src_height,
                              uchar* dst_data, size_t dst_step,
                              int dst_width, int dst_height,
                              double inv_scale_x, double inv_scale_y, int interpolation) {
    (void)src_type; (void)src_data; (void)src_step; (void)src_width; (void)src_height;
    (void)dst_data; (void)dst_step; (void)dst_width; (void)dst_height;
    (void)inv_scale_x; (void)inv_scale_y; (void)interpolation;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_warpAffine(int src_type,
                                    const uchar* src_data, size_t src_step,
                                    int src_width, int src_height,
                                    uchar* dst_data, size_t dst_step,
                                    int dst_width, int dst_height,
                                    const double M[6], int interpolation,
                                    int borderType, const double borderValue[4]) {
    (void)src_type; (void)src_data; (void)src_step; (void)src_width; (void)src_height;
    (void)dst_data; (void)dst_step; (void)dst_width; (void)dst_height;
    (void)M; (void)interpolation; (void)borderType; (void)borderValue;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_warpPerspective(int src_type,
                                       const uchar* src_data, size_t src_step,
                                       int src_width, int src_height,
                                       uchar* dst_data, size_t dst_step,
                                       int dst_width, int dst_height,
                                       const double M[9], int interpolation,
                                       int borderType, const double borderValue[4]) {
    (void)src_type; (void)src_data; (void)src_step; (void)src_width; (void)src_height;
    (void)dst_data; (void)dst_step; (void)dst_width; (void)dst_height;
    (void)M; (void)interpolation; (void)borderType; (void)borderValue;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_flip(int src_type,
                              const uchar* src_data, size_t src_step,
                              int src_width, int src_height,
                              uchar* dst_data, size_t dst_step,
                              int flip_mode) {
    (void)src_type; (void)src_data; (void)src_step; (void)src_width; (void)src_height;
    (void)dst_data; (void)dst_step; (void)flip_mode;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

// =========================================================================
// COLOR CONVERSIONS
// =========================================================================

extern "C" int rpp_hal_cvtBGRtoBGR(const uchar* src_data, size_t src_step,
                                   uchar* dst_data, size_t dst_step,
                                   int width, int height, int depth,
                                   int scn, int dcn, bool swapBlue) {
    (void)src_data; (void)src_step; (void)dst_data; (void)dst_step;
    (void)width; (void)height; (void)depth; (void)scn; (void)dcn; (void)swapBlue;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cvtBGRtoGray(const uchar* src_data, size_t src_step,
                                    uchar* dst_data, size_t dst_step,
                                    int width, int height, int depth,
                                    int scn, bool swapBlue) {
    (void)src_data; (void)src_step; (void)dst_data; (void)dst_step;
    (void)width; (void)height; (void)depth; (void)scn; (void)swapBlue;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cvtGraytoBGR(const uchar* src_data, size_t src_step,
                                    uchar* dst_data, size_t dst_step,
                                    int width, int height, int depth, int dcn) {
    (void)src_data; (void)src_step; (void)dst_data; (void)dst_step;
    (void)width; (void)height; (void)depth; (void)dcn;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cvtBGRtoHSV(const uchar* src_data, size_t src_step,
                                   uchar* dst_data, size_t dst_step,
                                   int width, int height, int depth,
                                   int scn, bool swapBlue, bool isFullRange, bool isHSV) {
    (void)src_data; (void)src_step; (void)dst_data; (void)dst_step;
    (void)width; (void)height; (void)depth; (void)scn; (void)swapBlue;
    (void)isFullRange; (void)isHSV;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cvtHSVtoBGR(const uchar* src_data, size_t src_step,
                                   uchar* dst_data, size_t dst_step,
                                   int width, int height, int depth,
                                   int dcn, bool swapBlue, bool isFullRange, bool isHSV) {
    (void)src_data; (void)src_step; (void)dst_data; (void)dst_step;
    (void)width; (void)height; (void)depth; (void)dcn; (void)swapBlue;
    (void)isFullRange; (void)isHSV;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}
