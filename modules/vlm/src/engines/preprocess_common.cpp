// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "preprocess_common.hpp"

namespace cv { namespace vlm {

void toNormalizedPlanes(const Mat& imageBgr, float rescale, const Vec3f& mean, const Vec3f& std_,
                        Mat dst[3])
{
    CV_CheckTypeEQ(imageBgr.type(), CV_8UC3, "vlm: expected an 8-bit BGR image");
    for (int c = 0; c < 3; c++)
        CV_CheckNE((double)std_[c], 0.0, "vlm: image_std must be non-zero");

    Mat rgb;
    cvtColor(imageBgr, rgb, COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, rescale);

    std::vector<Mat> planes(3);
    split(rgb, planes);
    for (int c = 0; c < 3; c++)
        planes[c].convertTo(dst[c], CV_32F, 1.0 / std_[c], -mean[c] / std_[c]);
}

}} // namespace cv::vlm
