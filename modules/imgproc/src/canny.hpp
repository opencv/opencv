// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.

#ifndef OPENCV_IMGPROC_CANNY_HPP
#define OPENCV_IMGPROC_CANNY_HPP

#include <deque>

namespace cv {

int canny_simd_width();

void canny_calc_magnitude(const short* _dx, const short* _dy, int* _mag_n,
                          int width, bool L2gradient);
void canny_nms_row(const int* _mag_a, const int* _mag_p, const int* _mag_n,
                   const short* _dx, const short* _dy, uchar* _pmap,
                   int width, int low, int high, std::deque<uchar*>& stack);
void canny_finalize_row(const uchar* pmap, uchar* pdst, int width);

} // namespace cv

#endif // OPENCV_IMGPROC_CANNY_HPP
