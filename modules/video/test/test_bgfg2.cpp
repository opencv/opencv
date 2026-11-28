// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/core/ocl.hpp"
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
// A learning rate of zero freezes the model: the frames are still classified against it, but no
// sample is updated. The freeze is implemented twice - the CPU invoker skips the update step and
// the OpenCL kernel is given counters that no stored update point can match - so both paths are
// covered here.
static void checkZeroLearningRate(int cn, bool useOCL)
{
    const int type = CV_MAKETYPE(CV_8U, cn);
    Ptr<BackgroundSubtractorKNN> knn = createBackgroundSubtractorKNN(500, 400.0, true);

    const Mat trainHost(64, 64, type, Scalar::all(40));
    const Mat otherHost(64, 64, type, Scalar::all(200));

    // apply() and getBackgroundImage() only take the OpenCL path when the arrays are UMats.
    UMat trainUMat, otherUMat;
    if (useOCL)
    {
        trainUMat = trainHost.getUMat(ACCESS_READ);
        otherUMat = otherHost.getUMat(ACCESS_READ);
    }

    Mat fgmaskHost;
    UMat fgmaskUMat;

    for (int i = 0; i < 30; i++)
    {
        if (useOCL)
            knn->apply(trainUMat, fgmaskUMat, 0.5);
        else
            knn->apply(trainHost, fgmaskHost, 0.5);
    }

    if (useOCL)
        fgmaskUMat.copyTo(fgmaskHost);
    EXPECT_EQ(0.0, cvtest::norm(fgmaskHost, NORM_INF)) << "the trained frame is background";

    Mat backgroundBefore;
    if (useOCL)
    {
        UMat backgroundUMat;
        knn->getBackgroundImage(backgroundUMat);
        backgroundUMat.copyTo(backgroundBefore);
    }
    else
        knn->getBackgroundImage(backgroundBefore);

    // The unrelated frames are still classified against the frozen model, so they are foreground,
    // but the background image must not move at all.
    for (int i = 0; i < 30; i++)
    {
        if (useOCL)
            knn->apply(otherUMat, fgmaskUMat, 0.0);
        else
            knn->apply(otherHost, fgmaskHost, 0.0);
    }

    if (useOCL)
        fgmaskUMat.copyTo(fgmaskHost);
    EXPECT_EQ(fgmaskHost.total(), (size_t)countNonZero(fgmaskHost))
        << "the unrelated frames are still classified as foreground";

    Mat backgroundAfter;
    if (useOCL)
    {
        UMat backgroundUMat;
        knn->getBackgroundImage(backgroundUMat);
        backgroundUMat.copyTo(backgroundAfter);
    }
    else
        knn->getBackgroundImage(backgroundAfter);

    Mat diff;
    absdiff(backgroundBefore, backgroundAfter, diff);
    EXPECT_DOUBLE_EQ(0.0, cv::norm(diff, NORM_INF)) << "a zero learning rate must not update the model";
}

TEST(BackgroundSubtractorKNN, ZeroLearningRateFreezesModel)
{
    const int channels[] = {1, 3};
    for (int i = 0; i < 2; i++)
        checkZeroLearningRate(channels[i], false);
}

TEST(BackgroundSubtractorKNN, ZeroLearningRateFreezesModel_OpenCL)
{
    if (!cv::ocl::useOpenCL())
        throw SkipTestException("OpenCL is not available / disabled");

    const int channels[] = {1, 3};
    for (int i = 0; i < 2; i++)
        checkZeroLearningRate(channels[i], true);
}

}} // namespace
/* End of file. */
