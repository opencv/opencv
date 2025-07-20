// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/video/background_segm.hpp"

namespace opencv_test { namespace {

using namespace cv;

class CV_MOG2Test : public cvtest::BaseTest
{
public:
    // CV_MOG2Test();
protected:
    // void SetUp() override {}
    // void TearDown() override {}

    Mat vid;
};

///////////////////////// MOG2 //////////////////////////////
TEST(BackgroundSubtractorMOG2, KnownForegroundMaskShadowsTrue)
{
    Ptr<BackgroundSubtractorMOG2> mog2 = createBackgroundSubtractorMOG2(500, 16, true);

    //Black Frame
    Mat input = Mat::zeros(480,640 , CV_8UC3);

    //White Rectangle
    Mat knownFG = Mat::zeros(input.size(), CV_8U);

    rectangle(knownFG, Rect(3,3,8,8), Scalar(255,255,255), -1);

    Mat output;
    mog2->apply(input, output, knownFG);

    for(int y = 3; y < 8; y++){
        for (int x = 3; x < 8; x++){
            EXPECT_EQ(output.at<uchar>(y,x),255) << "Expected foreground at (" << x << "," << y << ")";
        }
    }
}

TEST(BackgroundSubtractorMOG2, KnownForegroundMaskShadowsFalse)
{
    Ptr<BackgroundSubtractorMOG2> mog2 = createBackgroundSubtractorMOG2(500, 16, false);

    //Black Frame
    Mat input = Mat::zeros(480,640 , CV_8UC3);

    //White Rectangle
    Mat knownFG = Mat::zeros(input.size(), CV_8U);

    rectangle(knownFG, Rect(3,3,5,5), Scalar(255,255,255), FILLED);

    Mat output;
    mog2->apply(input, output, knownFG);

    for(int y = 3; y < 8; y++){
        for (int x = 3; x < 8; x++){
            EXPECT_EQ(output.at<uchar>(y,x),255) << "Expected foreground at (" << x << "," << y << ")";
        }
    }
}

///////////////////////// KNN //////////////////////////////

TEST(BackgroundSubtractorKNN, KnownForegroundMaskShadowsTrue)
{
    Ptr<BackgroundSubtractorKNN> knn = createBackgroundSubtractorKNN(500, 400.0, true);

    //Black Frame
    Mat input = Mat::zeros(480,640 , CV_8UC3);

    //White Rectangle
    Mat knownFG = Mat::zeros(input.size(), CV_8U);

    rectangle(knownFG, Rect(3,3,5,5), Scalar(255,255,255), FILLED);

    Mat output;
    knn->apply(input, output, knownFG);

    for(int y = 3; y < 8; y++){
        for (int x = 3; x < 8; x++){
            EXPECT_EQ(output.at<uchar>(y,x),255) << "Expected foreground at (" << x << "," << y << ")";
        }
    }
}

TEST(BackgroundSubtractorKNN, KnownForegroundMaskShadowsFalse)
{
    Ptr<BackgroundSubtractorKNN> knn = createBackgroundSubtractorKNN(500, 400.0, false);

    //Black Frame
    Mat input = Mat::zeros(480,640 , CV_8UC3);

    //White Rectangle
    Mat knownFG = Mat::zeros(input.size(), CV_8U);

    rectangle(knownFG, Rect(3,3,5,5), Scalar(255,255,255), FILLED);

    Mat output;
    knn->apply(input, output, knownFG);

    for(int y = 3; y < 8; y++){
        for (int x = 3; x < 8; x++){
            EXPECT_EQ(output.at<uchar>(y,x),255) << "Expected foreground at (" << x << "," << y << ")";
        }
    }
}

}} // namespace
/* End of file. */
