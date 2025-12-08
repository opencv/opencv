// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <vector>

namespace opencv_test { namespace {

Mat CropMid(InputArray src, int w, int h)
{
    Mat mat = src.getMat();
    return mat(Rect(mat.cols / 2 - w / 2, mat.rows / 2 - h / 2, w, h));
}

Mat GenerateTestImage(Size size)
{
    Mat image = Mat::zeros(size.height * 2, size.width * 2, CV_32F);
    rectangle(image,
              Point(static_cast<int>(size.width * 0.1), static_cast<int>(size.height * 0.1)),
              Point(static_cast<int>(size.width * 0.9), static_cast<int>(size.height * 0.9)),
              Scalar(1),
              -1);
    return image;
}

void TestPhaseCorrelationIterative(const Size& size, const double maxShift)
{
    const auto iters = std::max(201., maxShift * 10 + 1);
    const Point2d shiftOffset(-maxShift * 0.5, -maxShift * 0.5);
    Mat image1 = GenerateTestImage(size);
    Mat crop1 = CropMid(image1, size.width, size.height);
    Mat image2 = image1.clone();

    std::vector<double> pcErrors;
    std::vector<double> ipcErrors;

    for (int i = 0; i < iters; ++i)
    {
        const auto shift =
            Point2d(maxShift * i / (iters - 1), maxShift * i / (iters - 1)) + shiftOffset;
        const Mat Tmat = (Mat_<double>(2, 3) << 1., 0., shift.x, 0., 1., shift.y);
        warpAffine(image1, image2, Tmat, image2.size());
        Mat crop2 = CropMid(image2, size.width, size.height);
        const auto ipcshift = phaseCorrelateIterative(crop1, crop2);
        const auto pcshift = phaseCorrelate(crop1, crop2);

        pcErrors.push_back(
            0.5 * std::abs(pcshift.x - shift.y) + 0.5 * std::abs(pcshift.y - shift.x));
        ipcErrors.push_back(
            0.5 * std::abs(ipcshift.x - shift.y) + 0.5 * std::abs(ipcshift.y - shift.x));

        // error should be low
        EXPECT_NEAR(ipcshift.x - shift.x, 0.0, 0.1);
        EXPECT_NEAR(ipcshift.y - shift.y, 0.0, 0.1);
    }

    cv::Scalar pcMean, pcStddev, ipcMean, ipcStddev;
    meanStdDev(ipcErrors, ipcMean, ipcStddev);
    meanStdDev(pcErrors, pcMean, pcStddev);

    // average error should be low
    ASSERT_LT(ipcMean[0], 0.03);
    // average error should be less than non-iterative average error
    ASSERT_LT(ipcMean[0], pcMean[0]);
    // error stddev should be less than non-iterative error stddev
    ASSERT_LT(ipcStddev[0], pcStddev[0]);
}


TEST(Imgproc_PhaseCorrelationIterative, 256x128_accuracy)
{
    TestPhaseCorrelationIterative(Size(256, 128), 1);
}

TEST(Imgproc_PhaseCorrelationIterative, 64x64_accuracy_shift_1)
{
    TestPhaseCorrelationIterative(Size(64, 64), 1);
}

TEST(Imgproc_PhaseCorrelationIterative, 64x64_accuracy_shift_16)
{
    TestPhaseCorrelationIterative(Size(64, 64), 16);
}

TEST(Imgproc_PhaseCorrelationIterative, 0x0_image)
{
    ASSERT_ANY_THROW(TestPhaseCorrelationIterative(Size(0, 0), 1));
}

TEST(Imgproc_PhaseCorrelationIterative, 1x1_image)
{
    ASSERT_ANY_THROW(TestPhaseCorrelationIterative(Size(1, 1), 1));
}

TEST(Imgproc_PhaseCorrelationIterative, accuracy_real_img)
{
    Mat img = imread(cvtest::TS::ptr()->get_data_path() + "shared/airplane.png", IMREAD_GRAYSCALE);
    if (img.empty())
        return;
    img.convertTo(img, CV_64FC1);

    const int xLen = 256;
    const int yLen = 256;
    const int xShift = 40;
    const int yShift = 14;

    Mat roi1 = img(Rect(xShift, yShift, xLen, yLen));
    Mat roi2 = img(Rect(0, 0, xLen, yLen));

    const Point2d ipcShift = phaseCorrelateIterative(roi1, roi2);

    ASSERT_NEAR(ipcShift.x, (double)xShift, 1.);
    ASSERT_NEAR(ipcShift.y, (double)yShift, 1.);
}

}}  // namespace opencv_test
