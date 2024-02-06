#ifndef OPENCV_NDSRVP_IMGPROC_HPP_INCLUDED
#define OPENCV_NDSRVP_IMGPROC_HPP_INCLUDED

int ndsrvp_integral(int depth, int sdepth, int sqdepth,
                    const uchar * src, size_t _srcstep,
                    uchar * sum, size_t _sumstep,
                    uchar * sqsum, size_t,
                    uchar * tilted, size_t,
                    int width, int height, int cn);

#undef cv_hal_integral
#define cv_hal_integral ndsrvp_integral

#endif
