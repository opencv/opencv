// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_NDSRVP_IMGPROC_HPP
#define OPENCV_NDSRVP_IMGPROC_HPP

struct cvhalFilter2D;

namespace cv {

namespace ndsrvp {

enum InterpolationMasks {
    INTER_BITS = 5,
    INTER_BITS2 = INTER_BITS * 2,
    INTER_TAB_SIZE = 1 << INTER_BITS,
    INTER_TAB_SIZE2 = INTER_TAB_SIZE * INTER_TAB_SIZE
};

// ################ integral ################

int integral(int depth, int sdepth, int sqdepth,
    const uchar* src, size_t _srcstep,
    uchar* sum, size_t _sumstep,
    uchar* sqsum, size_t,
    uchar* tilted, size_t,
    int width, int height, int cn);

#undef cv_hal_integral
#define cv_hal_integral (cv::ndsrvp::integral)

// ################ warpAffine ################

int warpAffineBlocklineNN(int *adelta, int *bdelta, short* xy, int X0, int Y0, int bw);

#undef cv_hal_warpAffineBlocklineNN
#define cv_hal_warpAffineBlocklineNN (cv::ndsrvp::warpAffineBlocklineNN)

int warpAffineBlockline(int *adelta, int *bdelta, short* xy, short* alpha, int X0, int Y0, int bw);

#undef cv_hal_warpAffineBlockline
#define cv_hal_warpAffineBlockline (cv::ndsrvp::warpAffineBlockline)

// ################ warpPerspective ################

int warpPerspectiveBlocklineNN(const double *M, short* xy, double X0, double Y0, double W0, int bw);

#undef cv_hal_warpPerspectiveBlocklineNN
#define cv_hal_warpPerspectiveBlocklineNN (cv::ndsrvp::warpPerspectiveBlocklineNN)

int warpPerspectiveBlockline(const double *M, short* xy, short* alpha, double X0, double Y0, double W0, int bw);

#undef cv_hal_warpPerspectiveBlockline
#define cv_hal_warpPerspectiveBlockline (cv::ndsrvp::warpPerspectiveBlockline)

// ################ remap ################

int remap32f(int src_type, const uchar *src_data, size_t src_step, int src_width, int src_height,
    uchar *dst_data, size_t dst_step, int dst_width, int dst_height, float* mapx, size_t mapx_step,
    float* mapy, size_t mapy_step, int interpolation, int border_type, const double border_value[4]);

#undef cv_hal_remap32f
#define cv_hal_remap32f (cv::ndsrvp::remap32f)

// ################ threshold ################

int threshold(const uchar* src_data, size_t src_step,
    uchar* dst_data, size_t dst_step,
    int width, int height, int depth, int cn,
    double thresh, double maxValue, int thresholdType);

#undef cv_hal_threshold
#define cv_hal_threshold (cv::ndsrvp::threshold)

// ################ filter ################

int filterInit(cvhalFilter2D **context,
    uchar *kernel_data, size_t kernel_step,
    int kernel_type, int kernel_width,
    int kernel_height, int max_width, int max_height,
    int src_type, int dst_type, int borderType,
    double delta, int anchor_x, int anchor_y,
    bool allowSubmatrix, bool allowInplace);

#undef cv_hal_filterInit
#define cv_hal_filterInit (cv::ndsrvp::filterInit)

int filter(cvhalFilter2D *context,
    const uchar *src_data, size_t src_step,
    uchar *dst_data, size_t dst_step,
    int width, int height,
    int full_width, int full_height,
    int offset_x, int offset_y);

#undef cv_hal_filter
#define cv_hal_filter (cv::ndsrvp::filter)

int filterFree(cvhalFilter2D *context);

#undef cv_hal_filterFree
#define cv_hal_filterFree (cv::ndsrvp::filterFree)

// ################ medianBlur ################

int medianBlur(const uchar* src_data, size_t src_step,
    uchar* dst_data, size_t dst_step,
    int width, int height, int depth, int cn, int ksize);

#undef cv_hal_medianBlur
#define cv_hal_medianBlur (cv::ndsrvp::medianBlur)

// ################ bilateralFilter ################

int bilateralFilter(const uchar* src_data, size_t src_step,
    uchar* dst_data, size_t dst_step, int width, int height, int depth,
    int cn, int d, double sigma_color, double sigma_space, int border_type);

#undef cv_hal_bilateralFilter
#define cv_hal_bilateralFilter (cv::ndsrvp::bilateralFilter)

} // namespace ndsrvp

} // namespace cv

#endif
