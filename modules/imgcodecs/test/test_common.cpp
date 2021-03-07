// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "test_precomp.hpp"
#include "test_common.hpp"

namespace opencv_test {

static
Mat generateTestImageBGR_()
{
    Size sz(640, 480);
    Mat result(sz, CV_8UC3, Scalar::all(0));

    const string fname = cvtest::findDataFile("../cv/shared/baboon.png");
    Mat image = imread(fname, IMREAD_COLOR);
    CV_Assert(!image.empty());
    CV_CheckEQ(image.size(), Size(512, 512), "");
    Rect roi((640-512) / 2, 0, 512, 480);
    image(Rect(0, 0, 512, 480)).copyTo(result(roi));
    result(Rect(0,  0, 5, 5)).setTo(Scalar(0, 0, 255));  // R
    result(Rect(5,  0, 5, 5)).setTo(Scalar(0, 255, 0));  // G
    result(Rect(10, 0, 5, 5)).setTo(Scalar(255, 0, 0));  // B
    result(Rect(0,  5, 5, 5)).setTo(Scalar(128, 128, 128));  // gray
    //imshow("test_image", result); waitKey();
    return result;
}
Mat generateTestImageBGR()
{
    static Mat image = generateTestImageBGR_();  // initialize once
    CV_Assert(!image.empty());
    return image;
}

static
Mat generateTestImageGrayscale_()
{
    Mat imageBGR = generateTestImageBGR();
    CV_Assert(!imageBGR.empty());

    Mat result;
    cvtColor(imageBGR, result, COLOR_BGR2GRAY);
    return result;
}
Mat generateTestImageGrayscale()
{
    static Mat image = generateTestImageGrayscale_();  // initialize once
    return image;
}

}  // namespace
