// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_RVV_HAL_IMGPROC_HPP
#define OPENCV_RVV_HAL_IMGPROC_HPP

struct cvhalFilter2D;

namespace cv { namespace rvv_hal { namespace imgproc {

#if CV_HAL_RVV_1P0_ENABLED

/* ############ imageMoments ############ */

int imageMoments(const uchar* src_data, size_t src_step, int src_type,
                 int width, int height, bool binary, double m[10]);

#undef cv_hal_imageMoments
#define cv_hal_imageMoments cv::rvv_hal::imgproc::imageMoments

/* ############ filter ############ */

int filterInit(cvhalFilter2D** context, uchar* kernel_data, size_t kernel_step, int kernel_type, int kernel_width, int kernel_height, int /*max_width*/, int /*max_height*/, int src_type, int dst_type, int borderType, double delta, int anchor_x, int anchor_y, bool /*allowSubmatrix*/, bool /*allowInplace*/);
int filter(cvhalFilter2D* context, uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height, int full_width, int full_height, int offset_x, int offset_y);
int filterFree(cvhalFilter2D* context);

#undef cv_hal_filterInit
#define cv_hal_filterInit cv::rvv_hal::imgproc::filterInit
#undef cv_hal_filter
#define cv_hal_filter cv::rvv_hal::imgproc::filter
#undef cv_hal_filterFree
#define cv_hal_filterFree cv::rvv_hal::imgproc::filterFree

/* ############ sepFilter ############ */

int sepFilterInit(cvhalFilter2D **context, int src_type, int dst_type, int kernel_type, uchar* kernelx_data, int kernelx_length, uchar* kernely_data, int kernely_length, int anchor_x, int anchor_y, double delta, int borderType);
int sepFilter(cvhalFilter2D *context, uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height, int full_width, int full_height, int offset_x, int offset_y);
int sepFilterFree(cvhalFilter2D* context);

#undef cv_hal_sepFilterInit
#define cv_hal_sepFilterInit cv::rvv_hal::imgproc::sepFilterInit
#undef cv_hal_sepFilter
#define cv_hal_sepFilter cv::rvv_hal::imgproc::sepFilter
#undef cv_hal_sepFilterFree
#define cv_hal_sepFilterFree cv::rvv_hal::imgproc::sepFilterFree

/* ############ morph ############ */

int morphInit(cvhalFilter2D** context, int operation, int src_type, int dst_type, int /*max_width*/, int /*max_height*/, int kernel_type, uchar* kernel_data, size_t kernel_step, int kernel_width, int kernel_height, int anchor_x, int anchor_y, int borderType, const double borderValue[4], int iterations, bool /*allowSubmatrix*/, bool /*allowInplace*/);
int morph(cvhalFilter2D* context, uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height, int src_full_width, int src_full_height, int src_roi_x, int src_roi_y, int /*dst_full_width*/, int /*dst_full_height*/, int /*dst_roi_x*/, int /*dst_roi_y*/);
int morphFree(cvhalFilter2D* context);

#undef cv_hal_morphInit
#undef cv_hal_morph
#undef cv_hal_morphFree
#define cv_hal_morphInit cv::rvv_hal::imgproc::morphInit
#define cv_hal_morph cv::rvv_hal::imgproc::morph
#define cv_hal_morphFree cv::rvv_hal::imgproc::morphFree

/* ############ gaussianBlur ############ */

int gaussianBlurBinomial(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height, int depth, int cn, size_t margin_left, size_t margin_top, size_t margin_right, size_t margin_bottom, size_t ksize, int border_type);

#undef cv_hal_gaussianBlurBinomial
#define cv_hal_gaussianBlurBinomial cv::rvv_hal::imgproc::gaussianBlurBinomial

/* ############ medianBlur ############ */

int medianBlur(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height, int depth, int cn, int ksize);

#undef cv_hal_medianBlur
#define cv_hal_medianBlur cv::rvv_hal::imgproc::medianBlur

/* ############ boxFilter ############ */

int boxFilter(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height, int src_depth, int dst_depth, int cn, int margin_left, int margin_top, int margin_right, int margin_bottom, size_t ksize_width, size_t ksize_height, int anchor_x, int anchor_y, bool normalize, int border_type);

#undef cv_hal_boxFilter
#define cv_hal_boxFilter cv::rvv_hal::imgproc::boxFilter

/* ############ bilateralFilter ############ */

int bilateralFilter(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step,
                    int width, int height, int depth, int cn, int d, double sigma_color,
                    double sigma_space, int border_type);

#undef cv_hal_bilateralFilter
#define cv_hal_bilateralFilter cv::rvv_hal::imgproc::bilateralFilter

/* ############ pyramid ############ */

int pyrDown(const uchar* src_data, size_t src_step, int src_width, int src_height, uchar* dst_data, size_t dst_step, int dst_width, int dst_height, int depth, int cn, int border_type);
int pyrUp(const uchar* src_data, size_t src_step, int src_width, int src_height, uchar* dst_data, size_t dst_step, int dst_width, int dst_height, int depth, int cn, int border_type);

#undef cv_hal_pyrdown
#define cv_hal_pyrdown cv::rvv_hal::imgproc::pyrDown
#undef cv_hal_pyrup
#define cv_hal_pyrup cv::rvv_hal::imgproc::pyrUp

/* ############ cvtColor ############ */

int cvtBGRtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int depth, int scn, int dcn, bool swapBlue);
int cvtGraytoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int depth, int dcn);
int cvtBGRtoGray(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int depth, int scn, bool swapBlue);
int cvtBGR5x5toBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int dcn, bool swapBlue, int greenBits);
int cvtBGRtoBGR5x5(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int scn, bool swapBlue, int greenBits);
int cvtBGR5x5toGray(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int greenBits);
int cvtGraytoBGR5x5(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int greenBits);
int cvtYUVtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int depth, int dcn, bool swapBlue, bool isCbCr);
int cvtBGRtoYUV(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int depth, int scn, bool swapBlue, bool isCbCr);
int cvtOnePlaneYUVtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int dst_width, int dst_height, int dcn, bool swapBlue, int uIdx, int yIdx);
int cvtTwoPlaneYUVtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int dst_width, int dst_height, int dcn, bool swapBlue, int uIdx);
int cvtThreePlaneYUVtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int dst_width, int dst_height, int dcn, bool swapBlue, int uIdx);
int cvtOnePlaneBGRtoYUV(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int scn, bool swapBlue, int uIdx, int yIdx);
int cvtBGRtoTwoPlaneYUV(const uchar * src_data, size_t src_step, uchar * y_data, size_t y_step, uchar * uv_data, size_t uv_step, int width, int height, int scn, bool swapBlue, int uIdx);
int cvtBGRtoThreePlaneYUV(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int scn, bool swapBlue, int uIdx);
int cvtHSVtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int depth, int dcn, bool swapBlue, bool isFullRange, bool isHSV);
int cvtBGRtoHSV(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int depth, int scn, bool swapBlue, bool isFullRange, bool isHSV);
int cvtXYZtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int depth, int dcn, bool swapBlue);
int cvtBGRtoXYZ(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int depth, int scn, bool swapBlue);
int cvtLabtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int depth, int dcn, bool swapBlue, bool isLab, bool srgb);
int cvtBGRtoLab(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int depth, int scn, bool swapBlue, bool isLab, bool srgb);

#undef cv_hal_cvtBGRtoBGR
#define cv_hal_cvtBGRtoBGR cv::rvv_hal::imgproc::cvtBGRtoBGR
#undef cv_hal_cvtGraytoBGR
#define cv_hal_cvtGraytoBGR cv::rvv_hal::imgproc::cvtGraytoBGR
#undef cv_hal_cvtBGRtoGray
#define cv_hal_cvtBGRtoGray cv::rvv_hal::imgproc::cvtBGRtoGray
#undef cv_hal_cvtBGR5x5toBGR
#define cv_hal_cvtBGR5x5toBGR cv::rvv_hal::imgproc::cvtBGR5x5toBGR
#undef cv_hal_cvtBGRtoBGR5x5
#define cv_hal_cvtBGRtoBGR5x5 cv::rvv_hal::imgproc::cvtBGRtoBGR5x5
#undef cv_hal_cvtBGR5x5toGray
#define cv_hal_cvtBGR5x5toGray cv::rvv_hal::imgproc::cvtBGR5x5toGray
#undef cv_hal_cvtGraytoBGR5x5
#define cv_hal_cvtGraytoBGR5x5 cv::rvv_hal::imgproc::cvtGraytoBGR5x5
#undef cv_hal_cvtYUVtoBGR
#define cv_hal_cvtYUVtoBGR cv::rvv_hal::imgproc::cvtYUVtoBGR
#undef cv_hal_cvtBGRtoYUV
#define cv_hal_cvtBGRtoYUV cv::rvv_hal::imgproc::cvtBGRtoYUV
#undef cv_hal_cvtOnePlaneYUVtoBGR
#define cv_hal_cvtOnePlaneYUVtoBGR cv::rvv_hal::imgproc::cvtOnePlaneYUVtoBGR
#undef cv_hal_cvtTwoPlaneYUVtoBGR
#define cv_hal_cvtTwoPlaneYUVtoBGR cv::rvv_hal::imgproc::cvtTwoPlaneYUVtoBGR
#undef cv_hal_cvtThreePlaneYUVtoBGR
#define cv_hal_cvtThreePlaneYUVtoBGR cv::rvv_hal::imgproc::cvtThreePlaneYUVtoBGR
#undef cv_hal_cvtOnePlaneBGRtoYUV
#define cv_hal_cvtOnePlaneBGRtoYUV cv::rvv_hal::imgproc::cvtOnePlaneBGRtoYUV
#undef cv_hal_cvtBGRtoTwoPlaneYUV
#define cv_hal_cvtBGRtoTwoPlaneYUV cv::rvv_hal::imgproc::cvtBGRtoTwoPlaneYUV
#undef cv_hal_cvtBGRtoThreePlaneYUV
#define cv_hal_cvtBGRtoThreePlaneYUV cv::rvv_hal::imgproc::cvtBGRtoThreePlaneYUV
#undef cv_hal_cvtHSVtoBGR
#define cv_hal_cvtHSVtoBGR cv::rvv_hal::imgproc::cvtHSVtoBGR
#undef cv_hal_cvtBGRtoHSV
#define cv_hal_cvtBGRtoHSV cv::rvv_hal::imgproc::cvtBGRtoHSV
#undef cv_hal_cvtXYZtoBGR
#define cv_hal_cvtXYZtoBGR cv::rvv_hal::imgproc::cvtXYZtoBGR
#undef cv_hal_cvtBGRtoXYZ
#define cv_hal_cvtBGRtoXYZ cv::rvv_hal::imgproc::cvtBGRtoXYZ
#undef cv_hal_cvtLabtoBGR
#define cv_hal_cvtLabtoBGR cv::rvv_hal::imgproc::cvtLabtoBGR
#undef cv_hal_cvtBGRtoLab
#define cv_hal_cvtBGRtoLab cv::rvv_hal::imgproc::cvtBGRtoLab

/* ############ warp ############ */

int remap32f(int src_type, const uchar *src_data, size_t src_step, int src_width, int src_height,
             uchar *dst_data, size_t dst_step, int dst_width, int dst_height,
             float* mapx, size_t mapx_step, float* mapy, size_t mapy_step,
             int interpolation, int border_type, const double border_value[4]);
int remap32fc2(int src_type, const uchar *src_data, size_t src_step, int src_width, int src_height,
               uchar *dst_data, size_t dst_step, int dst_width, int dst_height,
               float* map, size_t map_step, int interpolation, int border_type, const double border_value[4]);
int remap16s(int src_type, const uchar *src_data, size_t src_step, int src_width, int src_height,
             uchar *dst_data, size_t dst_step, int dst_width, int dst_height,
             short* mapx, size_t mapx_step, ushort* mapy, size_t mapy_step,
             int interpolation, int border_type, const double border_value[4]);

// BUG: https://github.com/opencv/opencv/issues/27279
// #undef cv_hal_remap32f
// #define cv_hal_remap32f cv::cv_hal_rvv::imgproc::remap32f
// #undef cv_hal_remap32fc2
// #define cv_hal_remap32fc2 cv::cv_hal_rvv::imgproc::remap32fc2
// #undef cv_hal_remap16s
// #define cv_hal_remap16s cv::cv_hal_rvv::imgproc::remap16s

int warpAffine(int src_type, const uchar *src_data, size_t src_step, int src_width, int src_height, uchar *dst_data, size_t dst_step, int dst_width, int dst_height, const double M[6], int interpolation, int borderType, const double borderValue[4]);
int warpPerspective(int src_type, const uchar *src_data, size_t src_step, int src_width, int src_height, uchar *dst_data, size_t dst_step, int dst_width, int dst_height, const double M[9], int interpolation, int borderType, const double borderValue[4]);

// BUG: https://github.com/opencv/opencv/issues/27280
//#undef cv_hal_warpAffine
//#define cv_hal_warpAffine cv::cv_hal_rvv::imgproc::warpAffine
//#undef cv_hal_warpPerspective
//#define cv_hal_warpPerspective cv::cv_hal_rvv::imgproc::warpPerspective

/* ############ threshold ############ */

int threshold(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height, int depth, int cn, double thresh, double maxValue, int thresholdType);
int threshold_otsu(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height, int depth, double maxValue, int thresholdType, double* thresh);
int adaptiveThreshold(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height, double maxValue, int adaptiveMethod, int thresholdType, int blockSize, double C);

// disabled since UI is fast enough, only called in threshold_otsu
// #undef cv_hal_threshold
// #define cv_hal_threshold cv::rvv_hal::imgproc::threshold
#undef cv_hal_threshold_otsu
#define cv_hal_threshold_otsu cv::rvv_hal::imgproc::threshold_otsu
#undef cv_hal_adaptiveThreshold
#define cv_hal_adaptiveThreshold cv::rvv_hal::imgproc::adaptiveThreshold

/* ############ histogram ############ */

int equalize_hist(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step, int width, int height);

#undef cv_hal_equalize_hist
#define cv_hal_equalize_hist cv::rvv_hal::imgproc::equalize_hist

int calc_hist(const uchar* src_data, size_t src_step, int src_type, int src_width, int src_height, float* hist_data, int hist_size, const float** ranges, bool uniform, bool accumulate);

#undef cv_hal_calcHist
#define cv_hal_calcHist cv::rvv_hal::imgproc::calc_hist

/* ############ resize ############ */

int resize(int src_type, const uchar *src_data, size_t src_step, int src_width, int src_height, uchar *dst_data, size_t dst_step, int dst_width, int dst_height, double inv_scale_x, double inv_scale_y, int interpolation);

#undef cv_hal_resize
#define cv_hal_resize cv::rvv_hal::imgproc::resize

/* ############ resize ############ */

int integral(int depth, int sdepth, int sqdepth,
             const uchar* src_data, size_t src_step,
             uchar* sum_data, size_t sum_step,
             uchar* sqsum_data, size_t sqsum_step,
             uchar* tilted_data, [[maybe_unused]] size_t tilted_step,
             int width, int height, int cn);

// Diasbled due to accuracy issue.
// Details see https://github.com/opencv/opencv/issues/27407.
//#undef cv_hal_integral
//#define cv_hal_integral cv::rvv_hal::imgproc::integral

#endif // CV_HAL_RVV_1P0_ENABLED

#if CV_HAL_RVV_071_ENABLED

int cvtBGRtoBGR(const uchar * src_data, size_t src_step, uchar * dst_data, size_t dst_step, int width, int height, int depth, int scn, int dcn, bool swapBlue);
#undef cv_hal_cvtBGRtoBGR
#define cv_hal_cvtBGRtoBGR cv::rvv_hal::imgproc::cvtBGRtoBGR

#endif // CV_HAL_RVV_071_ENABLED

}}} // cv::rvv_hal::imgproc

#endif // OPENCV_RVV_HAL_IMGPROC_HPP
