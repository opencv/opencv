// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/video/background_segm.hpp"

namespace opencv_test { namespace {

using namespace cv;

///////////////////////// MOG2 //////////////////////////////
TEST(BackgroundSubtractorMOG2, KnownForegroundMaskShadowsTrue)
{
    Ptr<BackgroundSubtractorMOG2> mog2 = createBackgroundSubtractorMOG2(500, 16, true);

    //Black Frame
    Mat input = Mat::zeros(480,640 , CV_8UC3);

    //White Rectangle
    Mat knownFG = Mat::zeros(input.size(), CV_8U);

    rectangle(knownFG, Rect(3,3,5,5), Scalar(255,255,255), -1);

    Mat output;
    mog2->apply(input, knownFG, output);

    for(int y = 3; y < 8; y++)
    {
        for (int x = 3; x < 8; x++){
            EXPECT_EQ(255,output.at<uchar>(y,x)) << "Expected foreground at (" << x << "," << y << ")";
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
    mog2->apply(input, knownFG, output);

    for(int y = 3; y < 8; y++)
    {
        for (int x = 3; x < 8; x++){
            EXPECT_EQ(255,output.at<uchar>(y,x)) << "Expected foreground at (" << x << "," << y << ")";
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
    knn->apply(input, knownFG, output);

    for(int y = 3; y < 8; y++)
    {
        for (int x = 3; x < 8; x++){
            EXPECT_EQ(255,output.at<uchar>(y,x)) << "Expected foreground at (" << x << "," << y << ")";
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
    knn->apply(input, knownFG, output);

    for(int y = 3; y < 8; y++)
    {
        for (int x = 3; x < 8; x++){
            EXPECT_EQ(255,output.at<uchar>(y,x)) << "Expected foreground at (" << x << "," << y << ")";
        }
    }
}

// https://github.com/opencv/opencv/issues/24226
TEST(BackgroundSubtractorKNN, ZeroLearningRateFreezesModel)
{
    const int channels[] = {1, 3};

    for (int channelIdx = 0; channelIdx < 2; channelIdx++)
    {
        const int cn = channels[channelIdx];
        const int type = CV_MAKETYPE(CV_8U, cn);
        Ptr<BackgroundSubtractorKNN> knn = createBackgroundSubtractorKNN(500, 400.0, true);

        Mat train(64, 64, type, Scalar::all(40));
        Mat other(64, 64, type, Scalar::all(200));
        Mat fgmask;

        for (int i = 0; i < 30; i++)
            knn->apply(train, fgmask, 0.5);

        Mat backgroundBefore;
        knn->getBackgroundImage(backgroundBefore);

        // A zero learning rate freezes the model: the unrelated frames are still detected as
        // foreground, but the background image must not move at all.
        for (int i = 0; i < 30; i++)
            knn->apply(other, fgmask, 0.0);

        EXPECT_EQ(255, fgmask.at<uchar>(10, 10));

        Mat backgroundAfter;
        knn->getBackgroundImage(backgroundAfter);

        Mat diff;
        absdiff(backgroundBefore, backgroundAfter, diff);
        EXPECT_DOUBLE_EQ(0.0, cvtest::norm(diff, NORM_INF));
    }
}

}} // namespace
/* End of file. */
