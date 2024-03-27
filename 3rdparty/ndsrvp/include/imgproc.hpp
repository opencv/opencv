#ifndef OPENCV_NDSRVP_IMGPROC_HPP_INCLUDED
#define OPENCV_NDSRVP_IMGPROC_HPP_INCLUDED

enum InterpolationMasks {
    INTER_BITS = 5,
    INTER_BITS2 = INTER_BITS * 2,
    INTER_TAB_SIZE = 1 << INTER_BITS,
    INTER_TAB_SIZE2 = INTER_TAB_SIZE * INTER_TAB_SIZE
};

int ndsrvp_integral(int depth, int sdepth, int sqdepth,
    const uchar* src, size_t _srcstep,
    uchar* sum, size_t _sumstep,
    uchar* sqsum, size_t,
    uchar* tilted, size_t,
    int width, int height, int cn);

#undef cv_hal_integral
#define cv_hal_integral (ndsrvp_integral)

namespace cv {
void remap(InputArray _src, OutputArray _dst,
    InputArray _map1, InputArray _map2,
    int interpolation, int borderType, const Scalar& borderValue);

int ndsrvp_warpAffine(int src_type,
    const uchar* src_data, size_t src_step, int src_width, int src_height,
    uchar* dst_data, size_t dst_step, int dst_width, int dst_height,
    const double M[6], int interpolation, int borderType, const double borderValue[4]);
}

#undef cv_hal_warpAffine
#define cv_hal_warpAffine (cv::ndsrvp_warpAffine)

#endif
