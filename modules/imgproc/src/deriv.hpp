// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.

#ifndef OPENCV_IMGPROC_DERIV_HPP
#define OPENCV_IMGPROC_DERIV_HPP

namespace cv {

void Sobel3x3(const uchar* src, size_t src_step, int srcRows, int srcCols,
              int rowStart, int rowEnd,
              short* dx, short* dy, size_t dst_step, int borderType);
void Sobel5x5(const uchar* src, size_t src_step, int srcRows, int srcCols,
              int rowStart, int rowEnd,
              short* dx, short* dy, size_t dst_step, int borderType);
void Sobel3x3f(const uchar* src, size_t src_step, int srcRows, int srcCols,
               int rowStart, int rowEnd,
               float* dx, float* dy, size_t dst_step, float scale, int borderType);
void Sobel5x5f(const uchar* src, size_t src_step, int srcRows, int srcCols,
               int rowStart, int rowEnd,
               float* dx, float* dy, size_t dst_step, float scale, int borderType);

} // namespace cv

#endif // OPENCV_IMGPROC_DERIV_HPP
