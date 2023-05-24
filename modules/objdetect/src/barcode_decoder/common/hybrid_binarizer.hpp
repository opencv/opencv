// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __OPENCV_BARCODE_HYBRID_BINARIZER_HPP__
#define __OPENCV_BARCODE_HYBRID_BINARIZER_HPP__

namespace cv {
namespace barcode {

void hybridBinarization(const Mat &src, Mat &dst);

void
calculateThresholdForBlock(const std::vector<uchar> &luminances, int sub_width, int sub_height, int width, int height,
                           const Mat &black_points, Mat &dst);

Mat calculateBlackPoints(std::vector<uchar> luminances, int sub_width, int sub_height, int width, int height);
}
}
#endif //__OPENCV_BARCODE_HYBRID_BINARIZER_HPP__
