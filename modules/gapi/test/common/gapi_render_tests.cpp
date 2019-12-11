// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#include "../test_precomp.hpp"
#include "gapi_render_tests.hpp"

namespace opencv_test
{

cv::Scalar cvtBGRToYUVC(const cv::Scalar& bgr)
{
    double y = bgr[2] *  0.299000 + bgr[1] *  0.587000 + bgr[0] *  0.114000;
    double u = bgr[2] * -0.168736 + bgr[1] * -0.331264 + bgr[0] *  0.500000 + 128;
    double v = bgr[2] *  0.500000 + bgr[1] * -0.418688 + bgr[0] * -0.081312 + 128;
    return {y, u, v};
}

void drawMosaicRef(const cv::Mat& mat, const cv::Rect &rect, int cellSz)
{
    cv::Mat msc_roi = mat(rect);
    int crop_x = msc_roi.cols - msc_roi.cols % cellSz;
    int crop_y = msc_roi.rows - msc_roi.rows % cellSz;

    for(int i = 0; i < crop_y; i += cellSz ) {
        for(int j = 0; j < crop_x; j += cellSz) {
            auto cell_roi = msc_roi(cv::Rect(j, i, cellSz, cellSz));
            cell_roi = cv::mean(cell_roi);
        }
    }
}

void blendImageRef(cv::Mat& mat, const cv::Point& org, const cv::Mat& img, const cv::Mat& alpha)
{
    auto roi = mat(cv::Rect(org, img.size()));
    cv::Mat img32f_w;
    cv::merge(std::vector<cv::Mat>(3, alpha), img32f_w);

    cv::Mat roi32f_w(roi.size(), CV_32FC3, cv::Scalar::all(1.0));
    roi32f_w -= img32f_w;

    cv::Mat img32f, roi32f;
    img.convertTo(img32f, CV_32F, 1.0/255);
    roi.convertTo(roi32f, CV_32F, 1.0/255);

    cv::multiply(img32f, img32f_w, img32f);
    cv::multiply(roi32f, roi32f_w, roi32f);
    roi32f += img32f;

    roi32f.convertTo(roi, CV_8U, 255.0);
};

} // namespace opencv_test
