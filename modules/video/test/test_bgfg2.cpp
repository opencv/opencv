// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/core/ocl.hpp"

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

#ifdef HAVE_OPENCL
TEST(BackgroundSubtractorKNN, OclUcharSampleParity)
{
    if (!cv::ocl::haveOpenCL() || !cv::ocl::useOpenCL() || cv::ocl::Device::getDefault().isIntel())
        throw SkipTestException("Active non-Intel OpenCL device required");

    const Size sz(163, 121);
    const int types[] = { CV_8UC1, CV_8UC3 };
    const bool shadows[] = { false, true };

    for (int t : types)
    {
        for (bool sh : shadows)
        {
            Ptr<BackgroundSubtractorKNN> knnCpu = createBackgroundSubtractorKNN(500, 400.0, sh);
            Ptr<BackgroundSubtractorKNN> knnOcl = createBackgroundSubtractorKNN(500, 400.0, sh);

            Mat bg(sz, t);
            RNG rng(42 + t);
            rng.fill(bg, RNG::UNIFORM, Scalar::all(20), Scalar::all(80));

            Mat fgCpu, bgCpu;
            UMat fgOcl, bgOcl;

            for (int f = 0; f < 16; ++f)
            {
                Mat frame = bg.clone();
                Rect box(20 + f * 4, 25, 24, 24);
                rectangle(frame, box, Scalar(220, 200, 240), FILLED);
                if (sh)
                {
                    Rect shadowBox(20 + f * 4, 52, 20, 12);
                    frame(shadowBox) *= 0.65;
                }

                knnCpu->apply(frame, fgCpu, f < 8 ? 0.25 : 0.05);
                knnOcl->apply(frame, fgOcl, f < 8 ? 0.25 : 0.05);

                Mat fgOclHost = fgOcl.getMat(ACCESS_READ);
                EXPECT_EQ(sz, fgOclHost.size());

                // Compare foreground vs static background away from the moving box boundary
                if (f >= 10)
                {
                    Rect staticRoi(0, 0, 15, sz.height);
                    EXPECT_EQ(0.0, cv::norm(fgCpu(staticRoi), fgOclHost(staticRoi), NORM_INF))
                        << "t=" << t << " sh=" << sh << " f=" << f;
                    Rect fgCore(box.x + 2, box.y + 2, box.width - 4, box.height - 4);
                    EXPECT_GE(cv::countNonZero(fgOclHost(fgCore) == 255), fgCore.area() * 9 / 10)
                        << "t=" << t << " sh=" << sh << " f=" << f;
                }
            }

            knnCpu->getBackgroundImage(bgCpu);
            knnOcl->getBackgroundImage(bgOcl);
            Mat bgOclHost = bgOcl.getMat(ACCESS_READ);
            ASSERT_EQ(bgCpu.size(), bgOclHost.size());
            ASSERT_EQ(bgCpu.type(), bgOclHost.type());
            double meanDiff = cv::mean(cv::abs(bgCpu - bgOclHost))[0];
            EXPECT_LE(meanDiff, 2.0) << "t=" << t << " sh=" << sh;
        }
    }
}
#endif

}} // namespace
/* End of file. */
