// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef __IPP_HAL_IMGPROC_HPP__
#define __IPP_HAL_IMGPROC_HPP__

#include <opencv2/core/base.hpp>
#include "ipp_utils.hpp"

// Disabled in https://github.com/opencv/opencv/pull/13085 due large binary size
#define DISABLE_IPP_BOX_FILTER 1
// IPP filter2D integration is disabled in main OpenCV; kept behind a macro like box filter.
#define DISABLE_IPP_FILTER2D 1
// Too big difference compared to OpenCV FFT-based convolution, different results on masks > 7x7
#define IPP_DISABLE_FILTER2D_BIG_MASK 1

#if IPP_VERSION_X100 >= 810

#if defined(HAVE_IPP_IW)
int ipp_hal_warpAffine(int src_type, const uchar *src_data, size_t src_step, int src_width, int src_height, uchar *dst_data, size_t dst_step, int dst_width,
                       int dst_height, const double M[6], int interpolation, int borderType, const double borderValue[4]);

// Does not pass tests in 5.x branch
//#undef cv_hal_warpAffine
//#define cv_hal_warpAffine ipp_hal_warpAffine
#endif

#if IPP_VERSION_X100 >= 202600

int ipp_hal_warpPerspective(int src_type, const uchar *src_data, size_t src_step, int src_width, int src_height, uchar *dst_data, size_t dst_step, int dst_width,
                            int dst_height, const double M[9], int interpolation, int borderType, const double borderValue[4]);

// Does not pass tests in 5.x branch
//#undef cv_hal_warpPerspective
//#define cv_hal_warpPerspective ipp_hal_warpPerspective

#endif // IPP_VERSION_X100 >= 202600

int ipp_hal_remap32f(int src_type, const uchar *src_data, size_t src_step, int src_width, int src_height,
    uchar *dst_data, size_t dst_step, int dst_width, int dst_height,
    float* mapx, size_t mapx_step, float* mapy, size_t mapy_step,
    int interpolation, int border_type, const double border_value[4]);
#undef cv_hal_remap32f
#define cv_hal_remap32f ipp_hal_remap32f

#if defined(HAVE_IPP_IW) && !DISABLE_IPP_BOX_FILTER
int ipp_hal_boxFilter(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step,
                      int width, int height, int src_depth, int dst_depth, int cn,
                      int margin_left, int margin_top, int margin_right, int margin_bottom,
                      size_t ksize_width, size_t ksize_height, int anchor_x, int anchor_y,
                      bool normalize, int border_type);
#undef  cv_hal_boxFilter
#define cv_hal_boxFilter ipp_hal_boxFilter
#endif // defined(HAVE_IPP_IW) && !DISABLE_IPP_BOX_FILTER

#if defined(HAVE_IPP_IW) && !DISABLE_IPP_FILTER2D
int ipp_hal_filter2D(const uchar * src_data, size_t src_step, int src_type,
                     uchar * dst_data, size_t dst_step, int dst_type,
                     int width, int height, int full_width, int full_height,
                     int offset_x, int offset_y,
                     const uchar * kernel_data, size_t kernel_step, int kernel_type,
                     int kernel_width, int kernel_height,
                     int anchor_x, int anchor_y, double delta, int borderType,
                     bool isSubmatrix, bool allowInplace);
#undef cv_hal_filter_stateless
#define cv_hal_filter_stateless ipp_hal_filter2D
#endif // defined(HAVE_IPP_IW) && !DISABLE_IPP_FILTER2D

#endif //IPP_VERSION_X100 >= 810

#if IPP_VERSION_X100 >= 700

#define IPP_DISABLE_YUV_RGB 1 // accuracy difference
#define IPP_DISABLE_RGB_YUV 1 // breaks OCL accuracy tests
#define IPP_DISABLE_RGB_HSV 1 // breaks OCL accuracy tests
#define IPP_DISABLE_RGB_LAB 1 // breaks OCL accuracy tests
#define IPP_DISABLE_LAB_RGB 1 // breaks OCL accuracy tests
#define IPP_DISABLE_RGB_XYZ 1 // big accuracy difference
#define IPP_DISABLE_XYZ_RGB 1 // big accuracy difference

int ipp_hal_cvtBGRtoGray(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step,
                         int width, int height, int depth, int scn, bool swapBlue);
#undef cv_hal_cvtBGRtoGray
#define cv_hal_cvtBGRtoGray ipp_hal_cvtBGRtoGray

int ipp_hal_cvtBGRtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step,
                        int width, int height, int depth, int scn, int dcn, bool swapBlue);
#undef cv_hal_cvtBGRtoBGR
#define cv_hal_cvtBGRtoBGR ipp_hal_cvtBGRtoBGR

int ipp_hal_cvtGraytoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step,
                         int width, int height, int depth, int dcn);
#undef cv_hal_cvtGraytoBGR
#define cv_hal_cvtGraytoBGR ipp_hal_cvtGraytoBGR

int ipp_hal_cvtBGRtoHSV(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step,
                        int width, int height, int depth, int scn, bool swapBlue, bool isFullRange, bool isHSV);
#undef cv_hal_cvtBGRtoHSV
#define cv_hal_cvtBGRtoHSV ipp_hal_cvtBGRtoHSV

int ipp_hal_cvtHSVtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step,
                        int width, int height, int depth, int dcn, bool swapBlue, bool isFullRange, bool isHSV);
#undef cv_hal_cvtHSVtoBGR
#define cv_hal_cvtHSVtoBGR ipp_hal_cvtHSVtoBGR

int ipp_hal_cvtRGBAtoMultipliedRGBA(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step,
                                    int width, int height);
#undef cv_hal_cvtRGBAtoMultipliedRGBA
#define cv_hal_cvtRGBAtoMultipliedRGBA ipp_hal_cvtRGBAtoMultipliedRGBA

#if !IPP_DISABLE_RGB_YUV
int ipp_hal_cvtBGRtoYUV(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step,
                        int width, int height, int depth, int scn, bool swapBlue, bool isCbCr);
#undef cv_hal_cvtBGRtoYUV
#define cv_hal_cvtBGRtoYUV ipp_hal_cvtBGRtoYUV
#endif // !IPP_DISABLE_RGB_YUV

#if !IPP_DISABLE_YUV_RGB
int ipp_hal_cvtYUVtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step,
                        int width, int height, int depth, int dcn, bool swapBlue, bool isCbCr);
#undef cv_hal_cvtYUVtoBGR
#define cv_hal_cvtYUVtoBGR ipp_hal_cvtYUVtoBGR
#endif // !IPP_DISABLE_YUV_RGB

#if !IPP_DISABLE_RGB_XYZ
int ipp_hal_cvtBGRtoXYZ(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step,
                        int width, int height, int depth, int scn, bool swapBlue);
#undef cv_hal_cvtBGRtoXYZ
#define cv_hal_cvtBGRtoXYZ ipp_hal_cvtBGRtoXYZ
#endif // !IPP_DISABLE_RGB_XYZ

#if !IPP_DISABLE_XYZ_RGB
int ipp_hal_cvtXYZtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step,
                        int width, int height, int depth, int dcn, bool swapBlue);
#undef cv_hal_cvtXYZtoBGR
#define cv_hal_cvtXYZtoBGR ipp_hal_cvtXYZtoBGR
#endif // !IPP_DISABLE_XYZ_RGB

#if !IPP_DISABLE_RGB_LAB
int ipp_hal_cvtBGRtoLab(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step,
                        int width, int height, int depth, int scn, bool swapBlue, bool isLab, bool srgb);
#undef cv_hal_cvtBGRtoLab
#define cv_hal_cvtBGRtoLab ipp_hal_cvtBGRtoLab
#endif // !IPP_DISABLE_RGB_LAB

#if !IPP_DISABLE_LAB_RGB
int ipp_hal_cvtLabtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step,
                        int width, int height, int depth, int dcn, bool swapBlue, bool isLab, bool srgb);
#undef cv_hal_cvtLabtoBGR
#define cv_hal_cvtLabtoBGR ipp_hal_cvtLabtoBGR
#endif // !IPP_DISABLE_LAB_RGB

#endif //IPP_VERSION_X100 >= 700

#endif //__IPP_HAL_IMGPROC_HPP__
