// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

#ifndef __DETECTOR_ALIGN_HPP_
#define __DETECTOR_ALIGN_HPP_

#include <stdio.h>
#include <fstream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
using namespace cv;
using namespace std;

namespace cv {
class Align {
public:
    Align();
    Mat calcWarpMatrix(const Mat src, const Mat dst);
    std::vector<Point2f> warpBack(const std::vector<Point2f> &dst_pts);
    Mat crop(const Mat &inputImg, const Mat &srcPts, const float paddingW, const float paddingH,
             const int minPadding);

    void setRotate90(bool v) { rotate90_ = v; }

private:
    Mat crop(const Mat &inputImg, const int width, const int height);
    Mat M_;
    Mat M_inv_;

    int crop_x_;
    int crop_y_;
    bool rotate90_;
};
}  // namespace cv
#endif  // __DETECTOR_ALIGN_HPP_