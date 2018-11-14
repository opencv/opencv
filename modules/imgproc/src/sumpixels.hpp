// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019, Intel Corporation, all rights reserved.
#ifndef OPENCV_IMGPROC_SUM_PIXELS_HPP
#define OPENCV_IMGPROC_SUM_PIXELS_HPP

namespace cv
{

namespace opt_AVX512_SKX
{
#if CV_TRY_AVX512_SKX
    void calculate_integral_avx512(
            const uchar *src, size_t _srcstep,
            double *sum,      size_t _sumstep,
            double *sqsum,    size_t _sqsumstep,
            int width, int height, int cn);

#endif
} // end namespace opt_AVX512_SKX
} // end namespace cv

#endif
