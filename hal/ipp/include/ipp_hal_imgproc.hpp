// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef __IPP_HAL_IMGPROC_HPP__
#define __IPP_HAL_IMGPROC_HPP__

#include <opencv2/core/base.hpp>
#include "ipp_utils.hpp"

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

#endif //IPP_VERSION_X100 >= 810

#if IPP_VERSION_X100 >= 700
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

int ipp_hal_cvtBGRtoYUV(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step,
                        int width, int height, int depth, int scn, bool swapBlue, bool isCbCr);
#undef cv_hal_cvtBGRtoYUV
#define cv_hal_cvtBGRtoYUV ipp_hal_cvtBGRtoYUV

int ipp_hal_cvtYUVtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step,
                        int width, int height, int depth, int dcn, bool swapBlue, bool isCbCr);
#undef cv_hal_cvtYUVtoBGR
#define cv_hal_cvtYUVtoBGR ipp_hal_cvtYUVtoBGR

int ipp_hal_cvtBGRtoXYZ(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step,
                        int width, int height, int depth, int scn, bool swapBlue);
#undef cv_hal_cvtBGRtoXYZ
#define cv_hal_cvtBGRtoXYZ ipp_hal_cvtBGRtoXYZ

int ipp_hal_cvtXYZtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step,
                        int width, int height, int depth, int dcn, bool swapBlue);
#undef cv_hal_cvtXYZtoBGR
#define cv_hal_cvtXYZtoBGR ipp_hal_cvtXYZtoBGR

int ipp_hal_cvtBGRtoLab(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step,
                        int width, int height, int depth, int scn, bool swapBlue, bool isLab, bool srgb);
#undef cv_hal_cvtBGRtoLab
#define cv_hal_cvtBGRtoLab ipp_hal_cvtBGRtoLab

int ipp_hal_cvtLabtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step,
                        int width, int height, int depth, int dcn, bool swapBlue, bool isLab, bool srgb);
#undef cv_hal_cvtLabtoBGR
#define cv_hal_cvtLabtoBGR ipp_hal_cvtLabtoBGR

int ipp_hal_cvtColorYUV2Gray(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step,
                             int width, int height);
#undef cv_hal_cvtColorYUV2Gray
#define cv_hal_cvtColorYUV2Gray ipp_hal_cvtColorYUV2Gray
#endif //IPP_VERSION_X100 >= 700

#endif //__IPP_HAL_IMGPROC_HPP__
