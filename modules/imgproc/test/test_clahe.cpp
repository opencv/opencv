// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

typedef testing::TestWithParam<testing::tuple<MatType, Size, Size, double, bool> > CLAHE_Padding;

TEST_P(CLAHE_Padding, reflected_border)
{
    const int type = testing::get<0>(GetParam());
    const Size grid = testing::get<1>(GetParam());
    const Size extra = testing::get<2>(GetParam());
    const double clipLimit = testing::get<3>(GetParam());
    const bool useRoi = testing::get<4>(GetParam());
    const Size size(grid.width * 5 + extra.width, grid.height * 5 + extra.height);

    // The ROI ends at the parent's bottom-right corner so border extrapolation
    // uses the same pixels as a standalone image, while preserving a row stride.
    Mat storage(size.height + 2, size.width + 3, type);
    Mat src = useRoi ? storage(Rect(3, 2, size.width, size.height)) : Mat(size, type);
    theRNG().fill(src, RNG::UNIFORM, 0, type == CV_8UC1 ? 256 : 65536);

    const int bottom = (grid.height - size.height % grid.height) % grid.height;
    const int right = (grid.width - size.width % grid.width) % grid.width;
    Mat padded, expected, actual;
    cv::copyMakeBorder(src, padded, 0, bottom, 0, right, BORDER_REFLECT_101);

    Ptr<CLAHE> clahe = cv::createCLAHE(clipLimit, grid);
    clahe->apply(padded, expected);
    clahe->apply(src, actual);
    EXPECT_EQ(src.type(), actual.type());
    EXPECT_EQ(src.size(), actual.size());
    EXPECT_MAT_NEAR(expected(Rect(Point(), size)), actual, 0);

    Mat inPlace = src.clone();
    clahe->apply(inPlace, inPlace);
    EXPECT_MAT_NEAR(actual, inPlace, 0);
}

INSTANTIATE_TEST_CASE_P(Imgproc, CLAHE_Padding, testing::Combine(
    testing::Values(MatType(CV_8UC1), MatType(CV_16UC1)),
    testing::Values(Size(8, 8), Size(3, 5), Size(1, 8), Size(8, 1)),
    testing::Values(Size(0, 0), Size(1, 0), Size(0, 1), Size(1, 1)),
    testing::Values(0.0, 2.0),
    testing::Bool()));

}} // namespace
